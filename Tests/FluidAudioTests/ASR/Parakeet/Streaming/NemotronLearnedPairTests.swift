import XCTest

@testable import FluidAudio

/// Learned (original → term) pairs in the Nemotron biased decode: the
/// divergence-point push, near-completion acceptance and re-casing, against
/// a scripted toy "model" — no CoreML needed.
final class NemotronLearnedPairTests: XCTestCase {

    // MARK: - Toy vocabulary + scripted model

    private static let blank = 20
    private let pieces: [Int: String] = [
        0: "▁cl", 1: "oud", 2: "▁cloud", 3: "aude", 4: "▁claud", 5: "e", 6: "▁is",
        7: "▁C", 8: "la", 9: "ude", 10: ".", 11: "<en-US>", 12: "▁torvane",
    ]
    /// "Claude" spelled in its own casing: `▁C` `la` `ude`.
    private let casedClaude = [7, 8, 9]

    /// Logits per (frame, symbol index at that frame). A missing entry is a
    /// frame the model spends on blank. Every listed token gets its logit;
    /// every other token sits at -20, blank at 0 unless listed (-20 in a
    /// listed row that omits it).
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

    private func makeBias(
        terms: [String] = ["Claude"], pairs: [(String, String)]
    ) -> NemotronVocabularyBias {
        let bias = NemotronVocabularyBias(
            terms: terms.map { CustomVocabularyTerm(text: $0, weight: 4.5) }, pieces: pieces)!
        bias.setLearnedPairs(pairs.map { NemotronLearnedPair(original: $0.0, term: $0.1) })
        return bias
    }

    /// The manager's driver in miniature (one chunk, finish at the end).
    private func run(_ script: Script, bias: NemotronVocabularyBias) -> [Int] {
        let vocab = Self.blank + 1
        var ids: [Int] = []
        let decoder = Decoder(bias: bias, blankId: Self.blank, frame: 0, token: Self.blank, state: [])
        func apply(_ outcome: Decoder.Outcome) {
            switch outcome {
            case .blank: break
            case .emitted(let id, _): ids.append(id)
            case .rewound(let count, _, _): ids.removeSubrange(count...)
            case .replaced(let from, _, let tokens, _, let thenEmit):
                ids.removeSubrange(from...)
                ids.append(contentsOf: tokens)
                if let thenEmit { ids.append(thenEmit.tokenId) }
            }
        }
        func drive() {
            while decoder.needsStep(until: script.frames) {
                let row = script.logits(frame: decoder.frame, symbol: decoder.symbol, vocab: vocab)
                let best = row.indices.max { row[$0] < row[$1] }!
                apply(
                    decoder.consume(
                        NemotronStepLogits(count: vocab, argmax: best) { row[$0] },
                        stateOut: decoder.state + [decoder.token], idsCount: ids.count, timingsCount: ids.count))
            }
        }
        drive()
        while let outcome = decoder.resolveSpeculation(.finish) {
            apply(outcome)
            drive()
        }
        return ids
    }

    /// "▁cloud" is heard, "▁claud" is close; the model then barely wants
    /// the final "e" (`eLogit`).
    private func cloudScript(eLogit: Float, frames: Int = 2) -> Script {
        var script = Script(frames: frames)
        script.at(0, 0, [2: 5, 4: 0])
        script.at(0, 1, [Self.blank: 0, 5: eLogit])
        return script
    }

    // MARK: - Divergence push + re-casing

    func testPairFlipsTheOriginalAndRespellsTheTermInItsCasing() {
        let script = cloudScript(eLogit: -2)
        XCTAssertEqual(run(script, bias: makeBias(pairs: [])), [2], "the term bonus alone loses")
        // The pair flips "▁cloud" to "▁claud"; "e" completes the term, which
        // is re-spelled "▁C la ude" — never the case-folded "claudE".
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude)
    }

    func testPairFiresMidWordAtTheDivergingPiece() {
        // "▁cl" is shared; the model continues with "oud", the term with "aude".
        var script = Script(frames: 1)
        script.at(0, 0, [0: 5])
        script.at(0, 1, [1: 5, 3: -2])
        XCTAssertEqual(run(script, bias: makeBias(pairs: [])), [0, 1])
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude)
    }

    func testPairDoesNotReFireInTheWordAfterItsTerm() {
        var script = cloudScript(eLogit: -2)
        script.at(1, 0, [2: 5, 4: 0])  // "▁cloud" again, right after "Claude"
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude + [2])
    }

    // MARK: - Near-completion

    func testNearCompletionAtTheNextWord() {
        var script = cloudScript(eLogit: -10)  // the silent final e is lost
        script.at(1, 0, [6: 5, 5: -10])
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude + [6])
    }

    func testNearCompletionAtPunctuation() {
        var script = cloudScript(eLogit: -10)
        script.at(1, 0, [10: 5, 5: -10])
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude + [10])
    }

    func testNearCompletionAtTheEndOfTheUtterance() {
        let script = cloudScript(eLogit: -10, frames: 1)
        XCTAssertEqual(run(script, bias: makeBias(pairs: [("cloud", "Claude")])), casedClaude)
    }

    func testNearCompletionRules() {
        let bias = makeBias(terms: ["Claude", "cloud"], pairs: [("claud", "Claude")])
        func near(_ word: String) -> [Int]? {
            bias.nearCompletion(of: Array("▁\(word)".unicodeScalars))?.tokens
        }
        XCTAssertNil(near("claud"), "the pair's own original is never accepted")
        let other = makeBias(pairs: [("cloud", "Claude")])
        func nearOther(_ word: String) -> [Int]? {
            other.nearCompletion(of: Array("▁\(word)".unicodeScalars))?.tokens
        }
        XCTAssertEqual(nearOther("claud"), casedClaude, "one trailing letter short")
        XCTAssertNil(nearOther("clau"), "two letters short")
        XCTAssertNil(nearOther("claudx"), "not a prefix of the term")
        XCTAssertNil(nearOther("cloud"), "the original")
    }

    // MARK: - Configuration

    func testPairsNeedTheirTermInTheVocabularyAndSingleWords() {
        XCTAssertFalse(makeBias(terms: ["torvane"], pairs: [("cloud", "Claude")]).hasLearnedPairs)
        XCTAssertFalse(makeBias(pairs: [("a cloud", "Claude")]).hasLearnedPairs, "multi-word original")
        XCTAssertFalse(makeBias(pairs: [("cl", "Claude")]).hasLearnedPairs, "original below minTermLength")
        XCTAssertFalse(makeBias(pairs: [("CLAUDE", "Claude")]).hasLearnedPairs, "same word")
        XCTAssertTrue(makeBias(pairs: [("cloud", "Claude")]).hasLearnedPairs)
    }

    // MARK: - Timings

    func testRespelledTokensSpanTheReplacedWord() {
        let frame = ASRConstants.secondsPerEncoderFrame
        let replaced = [
            TokenTiming(token: "▁claud", tokenId: 4, startTime: 1 * frame, endTime: 2 * frame, confidence: 1),
            TokenTiming(token: "e", tokenId: 5, startTime: 4 * frame, endTime: 5 * frame, confidence: 1),
        ]
        let timings = StreamingNemotronMultilingualAsrManager.respelledTimings(
            casedClaude, replacing: replaced[...], wordEndFrame: nil
        ) { "\($0)" }
        XCTAssertEqual(timings.map(\.tokenId), casedClaude)
        XCTAssertEqual(timings.first!.startTime, 1 * frame, accuracy: 1e-9)
        XCTAssertEqual(timings.last!.endTime, 5 * frame, accuracy: 1e-9)
        for (a, b) in zip(timings, timings.dropFirst()) {
            XCTAssertEqual(a.endTime, b.startTime, accuracy: 1e-9)
        }
        // A piece without a timing yet extends the span through its frame.
        let extended = StreamingNemotronMultilingualAsrManager.respelledTimings(
            casedClaude, replacing: replaced[..<1], wordEndFrame: 7
        ) { "\($0)" }
        XCTAssertEqual(extended.last!.endTime, 8 * frame, accuracy: 1e-9)
    }
}
