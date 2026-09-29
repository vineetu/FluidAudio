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
/// Learned pairs (`NemotronVocabularyBias.learnedPairPush`) add a push at
/// each pair's divergence point on top of the term bonus; the flip it causes
/// opens a speculation like any other. A speculation a pair pushed has two
/// extra outcomes, both a `.replaced` rewrite of its word: kept, the word is
/// re-spelled in the term's own casing (the matcher is case-folded, so the
/// model may have written "claudE"); and when it reaches a word boundary —
/// the next word, word-ending punctuation, or the end of the utterance — one
/// letter short of the term, the term is committed instead of rewinding.
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
        /// Replace the accumulators from these lengths on with `tokens`,
        /// spread over the replaced word's time span (which runs through
        /// absolute `wordEndFrame` when set — a piece not appended yet), then
        /// append `thenEmit` if set. Decoding continues where it is.
        case replaced(
            idsFrom: Int, timingsFrom: Int, tokens: [Int], wordEndFrame: Int?,
            thenEmit: (tokenId: Int, frame: Int)?)
    }

    private struct Key: Hashable {
        let frame: Int
        let tokenCount: Int
    }

    /// The word being written: where it starts in the driver's accumulators
    /// and the ids written into it.
    private struct Word {
        let ids: Int
        let timings: Int
        var tokens: [Int]
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
        /// The word being written before the snapshot's step (restored on
        /// rewind) and the start of the word the first speculative piece
        /// lands in (nil when unknown).
        let word: Word?
        let speculationWordIds: Int?
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
    /// Nil until this decoder sees a word start: a session that begins
    /// mid-word (a rescue span) never rewrites text it did not decode.
    private var word: Word?
    /// The open speculation was opened or flipped by a learned pair.
    private var pairPushed = false
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
            if let push = bias.learnedPairPush(plain: plain, blankId: blankId, allowStart: allowStart) {
                var boosts = push.boosts
                for candidate in candidates where candidate.boost > boosts[candidate.tokenId, default: -.infinity] {
                    boosts[candidate.tokenId] = candidate.boost
                }
                choice = Self.pickWithPenalties(
                    plain: plain, boosts: boosts, penalties: push.penalties, count: logits.count,
                    logit: logits.logit)
                if choice != plain {
                    pairPushed = true
                    trace?("pair-push: '\(piece(plain))' -> '\(piece(choice))'")
                }
            } else if !candidates.isEmpty {
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
                        scalarPosition: bias.emittedScalarCount, key: key, word: word,
                        speculationWordIds: bias.isWordStart(choice) ? idsCount : word?.ids)
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
        let startsWord = bias.isWordStart(choice)
        let endsWord = startsWord || bias.isWordEndingPunctuation(choice)
        let finishedWord = word
        let finishedText = bias.writtenWord
        if startsWord {
            word = Word(ids: idsCount, timings: timingsCount, tokens: [choice])
        } else {
            word?.tokens.append(choice)
        }
        bias.observe(choice)

        if let open = speculation {
            let status = bias.speculationStatus(coveringScalar: open.scalarPosition)
            if status.completed {
                speculation = nil
                stats.kept += 1
                trace?("bias-keep: term completed at frame \(emittedAt)")
                if pairPushed, let current = word, current.ids == open.speculationWordIds,
                    let text = bias.writtenWord, let cased = bias.casedTermTokens(for: text),
                    cased != current.tokens, current.tokens.allSatisfy(bias.isText)
                {
                    pairPushed = false
                    word = Word(ids: current.ids, timings: current.timings, tokens: cased)
                    trace?("pair-recase at frame \(emittedAt)")
                    return .replaced(
                        idsFrom: current.ids, timingsFrom: current.timings, tokens: cased,
                        wordEndFrame: emittedAt, thenEmit: nil)
                }
                pairPushed = false
            } else if !status.alive {
                if pairPushed, endsWord, let written = finishedWord, written.ids == open.speculationWordIds,
                    written.tokens.allSatisfy(bias.isText), let text = finishedText,
                    let near = bias.nearCompletion(of: text)
                {
                    speculation = nil
                    pairPushed = false
                    stats.kept += 1
                    bias.renameWrittenWord(to: near.form, writtenLength: text.count, ended: startsWord)
                    let termEnd = written.ids + near.tokens.count
                    word =
                        startsWord
                        ? Word(ids: termEnd, timings: written.timings + near.tokens.count, tokens: [choice])
                        : Word(ids: written.ids, timings: written.timings, tokens: near.tokens + [choice])
                    trace?("pair-near-accept at frame \(emittedAt)")
                    return .replaced(
                        idsFrom: written.ids, timingsFrom: written.timings, tokens: near.tokens,
                        wordEndFrame: nil, thenEmit: (choice, emittedAt))
                }
                stats.divergedRewinds += 1
                return rewind()
            }
        }
        return .emitted(tokenId: choice, frame: emittedAt)
    }

    /// Argmax of `logit + boost - penalty`. When the plain argmax is itself
    /// penalized any token may win, so every id is scanned.
    static func pickWithPenalties(
        plain: Int, boosts: [Int: Float], penalties: [Int: Float], count: Int, logit: (Int) -> Float
    ) -> Int {
        var bestId = plain
        var bestScore = -Float.infinity
        if penalties[plain] == nil {
            bestScore = logit(plain)
            for (id, boost) in boosts where id < count {
                let score = logit(id) + boost - (penalties[id] ?? 0)
                if score > bestScore {
                    bestScore = score
                    bestId = id
                }
            }
            return bestId
        }
        for id in 0..<count {
            let score = logit(id) + (boosts[id] ?? 0) - (penalties[id] ?? 0)
            if score > bestScore {
                bestScore = score
                bestId = id
            }
        }
        return bestId
    }

    /// Undo the open speculation (no-op without one). Decoding then resumes
    /// from its snapshot with the start banned there.
    func resolveSpeculation(_ reason: ResolveReason) -> Outcome? {
        guard let open = speculation else { return nil }
        // The end of the utterance (or rescue span) ends the word too.
        if reason == .finish, pairPushed, let current = word, current.ids == open.speculationWordIds,
            current.tokens.allSatisfy(bias.isText), let text = bias.writtenWord,
            let near = bias.nearCompletion(of: text)
        {
            speculation = nil
            pairPushed = false
            stats.kept += 1
            bias.renameWrittenWord(to: near.form, writtenLength: text.count, ended: false)
            word = Word(ids: current.ids, timings: current.timings, tokens: near.tokens)
            trace?("pair-near-accept at the end of the utterance")
            return .replaced(
                idsFrom: current.ids, timingsFrom: current.timings, tokens: near.tokens, wordEndFrame: nil,
                thenEmit: nil)
        }
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
        pairPushed = false
        word = snapshot.word
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
