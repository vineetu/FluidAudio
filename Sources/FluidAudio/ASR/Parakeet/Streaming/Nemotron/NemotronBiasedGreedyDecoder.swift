import Foundation

/// Counters for one session of biased decoding (cleared by `reset()`).
public struct NemotronVocabularyBiasStats: Sendable, Equatable {
    /// Steps where a vocabulary piece overtook the plain argmax.
    public var flips = 0
    /// Speculations opened (the first flip of each biased stretch).
    public var speculations = 0
    /// Speculations that spelled a whole term and were kept.
    public var kept = 0
    /// Speculations undone, by reason. `rewinds` is their sum.
    public var rewinds = 0
    /// Of `rewinds`: the match died on an emitted token.
    public var divergedRewinds = 0
    /// Of `rewinds`: still open when its frames were about to be released.
    public var deadlineRewinds = 0
    /// Of `rewinds`: still open at the end of the utterance (or rescue span).
    public var finishRewinds = 0
    /// Of `rewinds`: closed so a blank-span rescue could run.
    public var rescueRewinds = 0
    /// Of `rewinds`: the snapshot lay in an earlier chunk than the one being
    /// decoded, so already-reported partial text was revised.
    public var crossChunkRewinds = 0
    /// True once the rewind budget ran out and biasing switched off.
    public var biasDisabled = false

    public init() {}
}

/// Logits of one decode step, read through a closure so every decode site
/// (contiguous Float32, strided Float16 batched joint, test tables) shares
/// one selection routine.
struct NemotronStepLogits {
    let count: Int
    /// Plain (unbiased) argmax.
    let argmax: Int
    let logit: (Int) -> Float
}

/// Greedy RNN-T decode with decode-time vocabulary bias, a start gate, and
/// speculation + rewind — a port of Jot for Windows' `PhraseBias` /
/// `ParakeetTranscriber.DecodeGreedy` onto the Nemotron streaming loop.
///
/// Three rules, each measured doing damage without it:
///
/// - **Start gate.** A term may *start* only where the plain argmax is a
///   non-blank, word-initial (`▁`) piece, outside a cooldown, and not at a
///   banned `(frame, tokenCount)` position. Biasing re-ranks which word is
///   written; it never writes a term out of silence or splits a word the
///   model is part-way through, and no second term opens while a
///   speculation is open. Continuations of an open match are not gated
///   (they may overtake blank) and get `continuationScale` × the bonus.
/// - **Speculation.** The first flip (biased pick != plain argmax) snapshots
///   the decoder: frame, symbol index, last token, LSTM state, accumulator
///   lengths, and the matcher's state. When a match containing the flipped
///   piece spells a whole term the snapshot is dropped (kept). When an
///   emitted token leaves no such match alive the decode is restored to the
///   snapshot, the failed match is abandoned, the position is banned, and
///   no new term may start until the next token is emitted (cooldown). So
///   each biased stretch outputs either the whole term or exactly what the
///   model would have written unbiased.
/// - **Budget.** After `maxRewinds` rewinds in one utterance biasing is
///   switched off (rewinding is not — an open speculation is still undone).
///
/// The decoder is a synchronous state machine: the driver asks
/// `needsStep(until:)`, runs the model for (`frame`, `symbol`, `token`,
/// `state`), and hands the logits to `consume`, applying the returned
/// `Outcome` to its token/timing accumulators. The frame cursor is
/// absolute, so a rewind may land in an earlier chunk as long as the driver
/// still holds that chunk's encoder frames — `speculationFrame` tells it
/// which frames must stay; `resolveSpeculation` undoes an open one on
/// demand (frames about to be released, end of utterance, blank rescue).
/// Every decision depends only on encoder frames and decoder state, never on
/// how finely audio was fed.
final class NemotronBiasedGreedyDecoder<State> {

    /// Windows' `MaxRollbacks`.
    static var defaultMaxRewinds: Int { 64 }

    enum ResolveReason {
        case deadline, finish, rescue
    }

    enum Outcome {
        /// Blank: the cursor moved to the next frame.
        case blank
        /// Append `tokenId`, emitted at absolute `frame`.
        case emitted(tokenId: Int, frame: Int)
        /// Truncate the accumulators to these lengths; decoding resumes at
        /// absolute `frame`.
        case rewound(idsCount: Int, timingsCount: Int, frame: Int)
    }

    private struct Key: Hashable {
        let frame: Int
        let tokenCount: Int
    }

    private struct Snapshot {
        let frame: Int
        let symbol: Int
        let token: Int
        let state: State
        let idsCount: Int
        let timingsCount: Int
        let match: NemotronVocabularyBias.MatchState
        /// Absolute scalar position of the first speculative piece.
        let scalarPosition: Int
        let key: Key
    }

    let bias: NemotronVocabularyBias
    let blankId: Int
    let maxSymbolsPerFrame: Int
    let maxRewinds: Int
    /// `FLUIDAUDIO_BIAS_LOG` sink; `piece` renders a token id.
    var trace: ((String) -> Void)?
    var piece: (Int) -> String = { "#\($0)" }

    /// Next frame to decode (absolute) and the number of tokens already
    /// emitted at it.
    private(set) var frame: Int
    private(set) var symbol = 0
    /// Last emitted token and the decoder state after it: the step input.
    var token: Int
    var state: State

    private var speculation: Snapshot?
    private var banned: Set<Key> = []
    private var cooldown = false
    private(set) var stats = NemotronVocabularyBiasStats()

    init(
        bias: NemotronVocabularyBias,
        blankId: Int,
        frame: Int,
        token: Int,
        state: State,
        maxSymbolsPerFrame: Int = 10,
        maxRewinds: Int = NemotronBiasedGreedyDecoder.defaultMaxRewinds,
        stats: NemotronVocabularyBiasStats = NemotronVocabularyBiasStats()
    ) {
        self.bias = bias
        self.blankId = blankId
        self.frame = frame
        self.token = token
        self.state = state
        self.maxSymbolsPerFrame = maxSymbolsPerFrame
        self.maxRewinds = maxRewinds
        self.stats = stats
    }

    /// Absolute frame of the open speculation's snapshot, if any. The driver
    /// must keep this frame's encoder output until the speculation resolves.
    var speculationFrame: Int? { speculation?.frame }

    var isSpeculating: Bool { speculation != nil }

    /// Move the cursor (e.g. over a chunk that was skipped without decoding).
    /// Only valid with no open speculation.
    func jump(to frame: Int) {
        precondition(speculation == nil, "jump with an open speculation")
        self.frame = frame
        symbol = 0
    }

    /// Whether another step is needed before the cursor reaches `endFrame`.
    func needsStep(until endFrame: Int) -> Bool {
        if symbol >= maxSymbolsPerFrame {
            frame += 1
            symbol = 0
        }
        return frame < endFrame
    }

    /// Consume one step's logits (computed for `frame`, `token`, `state`).
    /// `idsCount`/`timingsCount` are the driver's accumulator lengths now.
    func consume(
        _ logits: NemotronStepLogits, stateOut: State, idsCount: Int, timingsCount: Int
    ) -> Outcome {
        let plain = logits.argmax
        var choice = plain
        if !stats.biasDisabled {
            let key = Key(frame: frame, tokenCount: idsCount)
            // A fresh start while a speculation is open could only kill that
            // speculation's own match, so only continuations compete then.
            let allowStart =
                plain != blankId && bias.isWordStart(plain) && !cooldown && speculation == nil
                && !banned.contains(key)
            let candidates = bias.candidates(includeFreshStarts: allowStart)
            if !candidates.isEmpty {
                choice = StreamingNemotronMultilingualAsrManager.pickBiased(
                    plain: plain, candidates: candidates, count: logits.count, logit: logits.logit)
            }
            if choice != plain {
                stats.flips += 1
                trace?(
                    "bias-flip: '\(piece(plain))'(\(logits.logit(plain))) -> "
                        + "'\(piece(choice))'(\(logits.logit(choice)))")
                if speculation == nil {
                    speculation = Snapshot(
                        frame: frame, symbol: symbol, token: token, state: state,
                        idsCount: idsCount, timingsCount: timingsCount, match: bias.matchState,
                        scalarPosition: bias.emittedScalarCount, key: key)
                    stats.speculations += 1
                    trace?("bias-spec: open at frame \(frame) symbol \(symbol)")
                }
            }
        }

        guard choice != blankId else {
            frame += 1
            symbol = 0
            return .blank
        }

        let emittedAt = frame
        token = choice
        state = stateOut
        symbol += 1
        cooldown = false
        bias.observe(choice)

        if let open = speculation {
            let status = bias.speculationStatus(coveringScalar: open.scalarPosition)
            if status.completed {
                speculation = nil
                stats.kept += 1
                trace?("bias-keep: term completed at frame \(emittedAt)")
            } else if !status.alive {
                stats.divergedRewinds += 1
                return rewind()
            }
        }
        return .emitted(tokenId: choice, frame: emittedAt)
    }

    /// Undo the open speculation (no-op without one). Decoding then resumes
    /// from its snapshot with the start banned there.
    func resolveSpeculation(_ reason: ResolveReason) -> Outcome? {
        guard speculation != nil else { return nil }
        switch reason {
        case .deadline: stats.deadlineRewinds += 1
        case .finish: stats.finishRewinds += 1
        case .rescue: stats.rescueRewinds += 1
        }
        return rewind()
    }

    /// Take over counters accumulated elsewhere in the same utterance (a
    /// blank-span rescue's session), keeping the rewind budget shared.
    func adoptStats(_ other: NemotronVocabularyBiasStats) {
        stats = other
    }

    /// Record that the last rewind landed before `chunkStartFrame`.
    func noteCrossChunkRewind() {
        stats.crossChunkRewinds += 1
    }

    private func rewind() -> Outcome {
        guard let snapshot = speculation else { preconditionFailure("rewind without speculation") }
        speculation = nil
        frame = snapshot.frame
        symbol = snapshot.symbol
        token = snapshot.token
        state = snapshot.state
        // Abandon the failed match instead of restoring it: restored, it is
        // retried at the very same step and fails again until the budget
        // runs out (measured on Windows: 64 rewinds on 11 s).
        bias.matchState = snapshot.match
        bias.abandonMatches(before: snapshot.scalarPosition)
        banned.insert(snapshot.key)
        cooldown = true
        stats.rewinds += 1
        if stats.rewinds >= maxRewinds, !stats.biasDisabled {
            stats.biasDisabled = true
            trace?("bias-off: \(stats.rewinds) rewinds this utterance")
        }
        trace?("bias-rewind: back to frame \(snapshot.frame) symbol \(snapshot.symbol)")
        return .rewound(idsCount: snapshot.idsCount, timingsCount: snapshot.timingsCount, frame: snapshot.frame)
    }
}
