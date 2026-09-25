import XCTest

@testable import FluidAudio

/// Start gate + speculation/rewind for the Nemotron biased decode, against a
/// scripted toy "model" — no CoreML needed. The decoder state is the token
/// history, so every step also checks that a rewind restored the state the
/// step's inputs were computed from.
final class NemotronBiasedGreedyDecoderTests: XCTestCase {

    // MARK: - Toy vocabulary + scripted model

    private static let blank = 20
    private let pieces: [Int: String] = [
        0: "▁",
        1: "▁to", 2: "r", 3: "vane", 4: "▁tor", 5: "▁torvane",
        6: "▁the", 7: "▁ran", 8: "▁cr", 9: "an", 10: "▁okay", 11: "s",
        12: "<en-US>",
    ]

    /// Logits per (frame, symbol index at that frame). A missing entry is a
    /// frame the model spends on blank. Every listed token gets its logit;
    /// every other token sits at -20, blank at 0 unless listed.
    private struct Script {
        var steps: [Int: [Int: [Int: Float]]] = [:]
        let frames: Int

        mutating func at(_ frame: Int, _ symbol: Int = 0, _ logits: [Int: Float]) {
            steps[frame, default: [:]][symbol] = logits
        }

        func logits(frame: Int, symbol: Int, vocab: Int) -> [Float] {
            var out = [Float](repeating: -20, count: vocab)
            out[blank] = 0
            if let row = steps[frame]?[symbol] {
                out[blank] = row[blank] ?? -20
                for (token, value) in row { out[token] = value }
            }
            return out
        }
    }

    private typealias Decoder = NemotronBiasedGreedyDecoder<[Int]>

    private struct Result {
        var ids: [Int] = []
        var frames: [Int] = []
        /// Transcript after each chunk (what a partial callback would show).
        var partials: [[Int]] = []
        var stats = NemotronVocabularyBiasStats()
    }

    private func makeBias(_ terms: [String], boost: Float = 4.5) -> NemotronVocabularyBias {
        NemotronVocabularyBias(
            terms: terms.map { CustomVocabularyTerm(text: $0, weight: boost) }, pieces: pieces)!
    }

    /// Mirror of the manager's driver: decode `chunk` frames at a time, keep
    /// two chunks of frames (resolve a speculation older than that), resolve
    /// any open speculation at the end.
    private func run(
        _ script: Script, terms: [String]?, chunk: Int, maxRewinds: Int = 64
    ) -> Result {
        let vocab = Self.blank + 1
        var result = Result()
        guard let terms else {
            // Plain greedy reference.
            for frame in 0..<script.frames {
                var symbol = 0
                while symbol < 10 {
                    let row = script.logits(frame: frame, symbol: symbol, vocab: vocab)
                    let best = row.indices.max { row[$0] < row[$1] }!
                    if best == Self.blank { break }
                    result.ids.append(best)
                    result.frames.append(frame)
                    symbol += 1
                }
            }
            return result
        }
        let decoder = Decoder(
            bias: makeBias(terms), blankId: Self.blank, frame: 0, token: Self.blank, state: [],
            maxRewinds: maxRewinds)

        func apply(_ outcome: Decoder.Outcome) {
            switch outcome {
            case .blank: break
            case .emitted(let id, let frame):
                result.ids.append(id)
                result.frames.append(frame)
            case .rewound(let count, _, _):
                result.ids.removeSubrange(count...)
                result.frames.removeSubrange(count...)
            }
        }
        func drive(until end: Int) {
            while decoder.needsStep(until: end) {
                // The toy decoder state is the list of tokens it has consumed;
                // with the pending input token it must equal what was kept.
                XCTAssertEqual(
                    decoder.state + (decoder.token == Self.blank ? [] : [decoder.token]), result.ids,
                    "a rewind must restore the decoder state of the kept tokens")
                let row = script.logits(frame: decoder.frame, symbol: decoder.symbol, vocab: vocab)
                let best = row.indices.max { row[$0] < row[$1] }!
                let logits = NemotronStepLogits(count: vocab, argmax: best) { row[$0] }
                let consumed = decoder.state + (decoder.token == Self.blank ? [] : [decoder.token])
                apply(
                    decoder.consume(
                        logits, stateOut: consumed, idsCount: result.ids.count,
                        timingsCount: result.ids.count))
            }
        }

        var start = 0
        while start < script.frames {
            let end = min(start + chunk, script.frames)
            drive(until: end)
            // Same rule as the manager: frames before this chunk are released
            // when the next chunk arrives, so a speculation there is undone.
            while let frame = decoder.speculationFrame, frame < start,
                let outcome = decoder.resolveSpeculation(.deadline)
            {
                apply(outcome)
                drive(until: end)
            }
            result.partials.append(result.ids)
            start = end
        }
        while let outcome = decoder.resolveSpeculation(.finish) {
            apply(outcome)
            drive(until: script.frames)
        }
        result.stats = decoder.stats
        return result
    }

    // MARK: - Start gate

    func testNoTermOutOfSilence() {
        // The model says blank everywhere; "▁tor" is close but must not open
        // a term over a blank argmax.
        var script = Script(frames: 4)
        script.at(1, 0, [Self.blank: 1, 4: -1])
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(biased.ids, [])
        XCTAssertEqual(biased.stats.flips, 0)
    }

    func testNoTermStartMidWord() {
        // The model is mid-word ("an" after "▁cr"); a fresh "▁tor" start is
        // gated even though its boosted logit would win.
        var script = Script(frames: 4)
        script.at(0, 0, [8: 5])
        script.at(0, 1, [9: 2, 4: 1])
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(biased.ids, [8, 9])
        XCTAssertEqual(biased.stats.flips, 0)
    }

    // MARK: - Speculation

    /// "▁the" is heard, "▁tor" is close: bias flips, the next word breaks the
    /// match, and the rewind restores exactly the unbiased decode.
    private func divergingScript(flipFrame: Int = 1, breakFrame: Int = 2, frames: Int = 4) -> Script {
        var script = Script(frames: frames)
        script.at(flipFrame, 0, [6: 3, 4: 1])
        // After either token, the model's next word is "▁ran" (not "vane").
        script.at(breakFrame, 0, [7: 12, 3: -5])
        return script
    }

    func testDivergedMatchRewindsToUnbiasedOutput() {
        let script = divergingScript()
        let plain = run(script, terms: nil, chunk: 4)
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(plain.ids, [6, 7])
        XCTAssertEqual(biased.ids, plain.ids)
        XCTAssertEqual(biased.frames, plain.frames)
        XCTAssertEqual(biased.stats.flips, 1, "the banned start is not retried")
        XCTAssertEqual(biased.stats.rewinds, 1)
        XCTAssertEqual(biased.stats.divergedRewinds, 1)
        XCTAssertEqual(biased.stats.kept, 0)
    }

    func testCompletedTermIsKept() {
        var script = Script(frames: 4)
        script.at(1, 0, [6: 3, 4: 1])
        script.at(2, 0, [3: 4, 7: 3])  // "vane" wins on its own once "▁tor" is out
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(biased.ids, [4, 3])
        XCTAssertEqual(biased.stats.kept, 1)
        XCTAssertEqual(biased.stats.rewinds, 0)
    }

    func testContinuationMayOvertakeBlank() {
        var script = Script(frames: 4)
        script.at(1, 0, [6: 3, 4: 1])
        script.at(2, 0, [Self.blank: 2, 3: -1])  // "vane" + 4.5 beats blank
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(biased.ids, [4, 3])
        XCTAssertEqual(biased.stats.kept, 1)
    }

    func testUnrelatedCompletedTermDoesNotKeepADeadSpeculation() {
        // "▁tor" (biased) then the model says "▁okay", a whole *other* term.
        // That completion does not contain the speculative piece, so the
        // "▁tor" stretch is still undone.
        var script = Script(frames: 4)
        script.at(1, 0, [6: 3, 4: 1])
        script.at(2, 0, [10: 12])
        let biased = run(script, terms: ["torvane", "okay"], chunk: 4)
        XCTAssertEqual(biased.ids, [6, 10])
        XCTAssertEqual(biased.stats.rewinds, 1)
    }

    func testCooldownBlocksRestartUntilNextToken() {
        // Right after the rewind the same frame decodes unbiased even though
        // "▁cr"... any fresh start is blocked until a token is emitted.
        var script = divergingScript()
        script.at(1, 1, [Self.blank: 0])
        let biased = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(biased.ids, [6, 7])
    }

    func testRewindBudgetSwitchesBiasOff() {
        // Eight separate diverging opportunities, budget 3: after the third
        // rewind no further flip happens and the output is plain.
        var script = Script(frames: 16)
        for k in 0..<8 {
            script.at(2 * k, 0, [6: 3, 4: 1])
            script.at(2 * k + 1, 0, [7: 12])
        }
        let plain = run(script, terms: nil, chunk: 16)
        let biased = run(script, terms: ["torvane"], chunk: 16, maxRewinds: 3)
        XCTAssertEqual(biased.ids, plain.ids)
        XCTAssertEqual(biased.stats.rewinds, 3)
        XCTAssertEqual(biased.stats.flips, 3)
        XCTAssertTrue(biased.stats.biasDisabled)
    }

    // MARK: - Chunk edges and end of utterance

    func testCrossChunkRewindMatchesSinglePass() {
        // Flip in the last frame of chunk 0, divergence in chunk 1.
        let script = divergingScript(flipFrame: 3, breakFrame: 5, frames: 8)
        let whole = run(script, terms: ["torvane"], chunk: 8)
        let chunked = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(chunked.ids, whole.ids)
        XCTAssertEqual(chunked.frames, whole.frames)
        XCTAssertEqual(chunked.ids, run(script, terms: nil, chunk: 8).ids)
        // Chunk 0's partial showed the speculative "▁tor"; chunk 1 revised it.
        XCTAssertEqual(chunked.partials.first, [4])
        XCTAssertEqual(chunked.partials.last, [6, 7])
    }

    func testCrossChunkCompletionMatchesSinglePass() {
        var script = Script(frames: 8)
        script.at(3, 0, [6: 3, 4: 1])
        script.at(5, 0, [3: 4, 7: 3])
        let whole = run(script, terms: ["torvane"], chunk: 8)
        let chunked = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(whole.ids, [4, 3])
        XCTAssertEqual(chunked.ids, whole.ids)
        XCTAssertEqual(chunked.frames, whole.frames)
    }

    func testFeedGranularityDoesNotChangeDecisions() {
        // Every chunk size that keeps the speculation inside the retained
        // window decodes the same.
        let script = divergingScript(flipFrame: 5, breakFrame: 6, frames: 12)
        let reference = run(script, terms: ["torvane"], chunk: 12)
        for size in [3, 4, 6] {
            XCTAssertEqual(run(script, terms: ["torvane"], chunk: size).ids, reference.ids, "chunk \(size)")
        }
    }

    func testSpeculationOlderThanRetainedFramesIsUndone() {
        // "▁tor" flips at frame 1, then a long pause: still alive (blank
        // does not break a match) when its frames are about to be released.
        var script = Script(frames: 12)
        script.at(1, 0, [6: 3, 4: 1])
        script.at(10, 0, [3: 4, 7: 3])  // would complete, but too late
        let chunked = run(script, terms: ["torvane"], chunk: 4)
        XCTAssertEqual(chunked.stats.deadlineRewinds, 1)
        XCTAssertEqual(chunked.ids, [6, 3])
    }

    func testTermCutOffAtFinishIsUndone() {
        // The utterance ends right after the speculative "▁tor".
        var script = Script(frames: 4)
        script.at(3, 0, [6: 3, 4: 1])
        let biased = run(script, terms: ["torvane"], chunk: 2)
        XCTAssertEqual(biased.ids, [6])
        XCTAssertEqual(biased.stats.finishRewinds, 1)
    }

    func testNoMatchingTermIsIdenticalToPlain() {
        let script = divergingScript(flipFrame: 1, breakFrame: 2, frames: 6)
        XCTAssertEqual(run(script, terms: ["xylophone"], chunk: 2).ids, run(script, terms: nil, chunk: 6).ids)
    }

    // MARK: - Matcher support

    func testSpeculationStatusRequiresMatchContainingSpeculativePiece() throws {
        let bias = makeBias(["torvane", "okay"])
        bias.observe(6)  // ▁the
        let position = bias.emittedScalarCount
        bias.observe(4)  // ▁tor (speculative)
        XCTAssertEqual(bias.speculationStatus(coveringScalar: position).alive, true)
        XCTAssertEqual(bias.speculationStatus(coveringScalar: position).completed, false)
        bias.observe(3)  // vane
        XCTAssertEqual(bias.speculationStatus(coveringScalar: position).completed, true)
        bias.observe(10)  // ▁okay: a whole term, but not the speculative one
        XCTAssertEqual(bias.speculationStatus(coveringScalar: position).alive, false)
    }

    func testAbandonMatchesDropsContinuationsButAllowsNewMatches() {
        let bias = makeBias(["torvane"])
        bias.observe(4)  // ▁tor
        XCTAssertTrue(bias.candidates(includeFreshStarts: false).contains { $0.tokenId == 3 })
        bias.abandonMatches(before: bias.emittedScalarCount)
        XCTAssertTrue(bias.candidates(includeFreshStarts: false).isEmpty)
        bias.observe(4)  // a new ▁tor after the barrier opens again
        XCTAssertTrue(bias.candidates(includeFreshStarts: false).contains { $0.tokenId == 3 })
    }

    func testMatchStateRestoresTailAndBarrier() {
        let bias = makeBias(["torvane"])
        bias.observe(1)  // ▁to
        let saved = bias.matchState
        bias.observe(7)
        bias.abandonMatches(before: bias.emittedScalarCount)
        bias.matchState = saved
        XCTAssertTrue(bias.candidates(includeFreshStarts: false).contains { $0.tokenId == 2 })  // r
    }

    func testContinuationOnlyCandidatesExcludeFreshStarts() {
        let bias = makeBias(["torvane"])
        XCTAssertTrue(bias.candidates(includeFreshStarts: false).isEmpty)
        XCTAssertFalse(bias.candidates().isEmpty)
        XCTAssertTrue(bias.isWordStart(4))
        XCTAssertFalse(bias.isWordStart(3))
        XCTAssertFalse(bias.isWordStart(12), "lang tags are not text")
    }
}
