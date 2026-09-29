import Accelerate
@preconcurrency import CoreML
import Foundation

/// Decoder LSTM state for the biased decode (the `State` of
/// `NemotronBiasedGreedyDecoder`). `h_out`/`c_out` are loopback outputs and
/// never output-backed, so holding the references is a true snapshot.
struct NemotronDecoderState {
    let h: MLMultiArray
    let c: MLMultiArray
}

/// One chunk's encoder output, deep-copied into arrays this session owns.
/// The live `encoded` tensor is an output backing the encoder rewrites on its
/// next prediction (and the triple-stage prefetch runs that prediction while
/// this chunk decodes), so a rewind into an earlier chunk cannot read it.
struct NemotronRetainedChunk {
    let startFrame: Int
    let frameCount: Int
    /// [1, encoderDim, T] contiguous Float32.
    let encoded: MLMultiArray
    /// [1, T, jointDim] contiguous Float32, when the encoder emits it.
    let encoderProj: MLMultiArray?

    func contains(_ frame: Int) -> Bool {
        frame >= startFrame && frame < startFrame + frameCount
    }
}

/// Per-session state of the biased decode: the speculation state machine
/// plus the encoder frames a rewind may need.
final class NemotronBiasDecodeSession {
    let decoder: NemotronBiasedGreedyDecoder<NemotronDecoderState>
    var chunks: [NemotronRetainedChunk] = []
    /// Batched-joint logits of the speculative fast path, valid for one
    /// decoder input (`token` + the `h` object) over frames
    /// `[firstFrame, firstFrame + count)`.
    var batch: (h: MLMultiArray, token: Int, firstFrame: Int, count: Int, logits: MLMultiArray, state: NemotronDecoderState)?

    init(decoder: NemotronBiasedGreedyDecoder<NemotronDecoderState>) {
        self.decoder = decoder
    }

    var decodedEnd: Int { chunks.last.map { $0.startFrame + $0.frameCount } ?? decoder.frame }
}

extension StreamingNemotronMultilingualAsrManager {

    /// Chunks of encoder output kept for rewinds: the one being decoded and
    /// the one before it. A speculation whose snapshot would fall out of this
    /// window is undone first (`deadline`), so a speculation lives at most
    /// until the end of the chunk after the one it opened in. Chunk
    /// boundaries depend only on the audio, never on how it was fed.
    static var biasRetainedChunks: Int { 2 }

    /// Counters for the current (or most recently finished) utterance.
    public func vocabularyBiasStats() -> NemotronVocabularyBiasStats {
        biasSession?.decoder.stats ?? lastBiasStats
    }

    internal func makeBiasSession(
        bias: NemotronVocabularyBias, stats: NemotronVocabularyBiasStats
    ) throws -> NemotronBiasDecodeSession {
        guard let h = hState, let c = cState else { throw ASRError.notInitialized }
        let decoder = NemotronBiasedGreedyDecoder<NemotronDecoderState>(
            bias: bias, blankId: config.blankIdx, frame: absoluteFrameBase, token: Int(lastToken),
            state: NemotronDecoderState(h: h, c: c), stats: stats)
        if nemotronBiasLogEnabled {
            let tokenizer = self.tokenizer
            let blank = config.blankIdx
            decoder.trace = { FileHandle.standardError.write(Data(($0 + "\n").utf8)) }
            decoder.piece = { id in id == blank ? "<blank>" : (tokenizer?.rawToken(for: id) ?? "?") }
        }
        return NemotronBiasDecodeSession(decoder: decoder)
    }

    /// Decode one chunk with vocabulary bias. Returns whether the token
    /// stream changed (for the partial callback).
    internal func decodeChunkWithBias(
        encoded: MLMultiArray,
        encoderProj: MLMultiArray?,
        numEncoderFrames: Int,
        bias: NemotronVocabularyBias,
        tokenizer: NemotronMultilingualTokenizer
    ) async throws -> Bool {
        let session: NemotronBiasDecodeSession
        if let existing = biasSession, existing.decoder.bias === bias {
            session = existing
        } else {
            session = try makeBiasSession(bias: bias, stats: NemotronVocabularyBiasStats())
            biasSession = session
        }
        let chunkStart = absoluteFrameBase
        var changed = false

        if session.decoder.frame != chunkStart {
            // A gap (VAD-skipped chunk): settle what the kept frames can
            // still undo, then continue after the gap.
            changed = try await settleBiasSpeculation(.deadline, tokenizer: tokenizer) || changed
            session.decoder.jump(to: chunkStart)
            session.chunks.removeAll()
        }

        session.chunks.append(
            NemotronRetainedChunk(
                startFrame: chunkStart, frameCount: numEncoderFrames,
                encoded: try Self.contiguousFloat32Copy(encoded),
                encoderProj: try encoderProj.map(Self.contiguousFloat32Copy)))
        if session.chunks.count > Self.biasRetainedChunks {
            session.chunks.removeFirst(session.chunks.count - Self.biasRetainedChunks)
        }
        session.batch = nil

        let end = chunkStart + numEncoderFrames
        changed = try await driveBiasedDecode(session, until: end, tokenizer: tokenizer) || changed

        // The previous chunk's frames are released when the next chunk
        // arrives, so a speculation opened there is resolved now.
        while let frame = session.decoder.speculationFrame, frame < chunkStart,
            let outcome = session.decoder.resolveSpeculation(.deadline)
        {
            applyBiasOutcome(outcome, session: session, tokenizer: tokenizer)
            changed = true
            _ = try await driveBiasedDecode(session, until: end, tokenizer: tokenizer)
        }

        lastToken = Int32(session.decoder.token)
        hState = session.decoder.state.h
        cState = session.decoder.state.c
        return changed
    }

    /// Undo any open speculation and re-decode the kept frames, repeating
    /// until none is open. Used at end of utterance (the final text must not
    /// carry an unfinished term), before a blank-span rescue, and across a
    /// gap. Returns whether the token stream changed.
    @discardableResult
    internal func settleBiasSpeculation(
        _ reason: NemotronBiasedGreedyDecoder<NemotronDecoderState>.ResolveReason,
        tokenizer: NemotronMultilingualTokenizer
    ) async throws -> Bool {
        guard let session = biasSession else { return false }
        var changed = false
        let end = session.decodedEnd
        while let outcome = session.decoder.resolveSpeculation(reason) {
            applyBiasOutcome(outcome, session: session, tokenizer: tokenizer)
            changed = true
            _ = try await driveBiasedDecode(session, until: end, tokenizer: tokenizer)
        }
        if changed {
            lastToken = Int32(session.decoder.token)
            hState = session.decoder.state.h
            cState = session.decoder.state.c
        }
        return changed
    }

    private func driveBiasedDecode(
        _ session: NemotronBiasDecodeSession, until end: Int, tokenizer: NemotronMultilingualTokenizer
    ) async throws -> Bool {
        var changed = false
        let decoder = session.decoder
        while decoder.needsStep(until: end) {
            let (logits, stateOut) = try await biasedStep(session)
            let outcome = decoder.consume(
                logits, stateOut: stateOut, idsCount: accumulatedTokenIds.count,
                timingsCount: accumulatedTokenTimings.count)
            if case .blank = outcome { continue }
            applyBiasOutcome(outcome, session: session, tokenizer: tokenizer)
            changed = true
        }
        return changed
    }

    private func applyBiasOutcome(
        _ outcome: NemotronBiasedGreedyDecoder<NemotronDecoderState>.Outcome,
        session: NemotronBiasDecodeSession,
        tokenizer: NemotronMultilingualTokenizer
    ) {
        switch outcome {
        case .blank:
            break
        case .emitted(let tokenId, let frame):
            accumulatedTokenIds.append(tokenId)
            appendTokenTiming(
                tokenId, frameInChunk: frame - absoluteFrameBase, tokenizer: tokenizer, observeBias: false)
            if config.langTagTokenIds.contains(tokenId),
                let piece = tokenizerPiece(forId: tokenId, tokenizer: tokenizer)
            {
                recordDetectedLanguage(NemotronMultilingualTokenizer.stripAngleBrackets(piece))
            }
        case .replaced(let idsFrom, let timingsFrom, let tokens, let wordEndFrame, let thenEmit):
            let timings = Self.respelledTimings(
                tokens, replacing: accumulatedTokenTimings[timingsFrom...], wordEndFrame: wordEndFrame
            ) { tokenizer.rawToken(for: $0) ?? "" }
            accumulatedTokenIds.removeSubrange(idsFrom...)
            accumulatedTokenTimings.removeSubrange(timingsFrom...)
            accumulatedTokenIds.append(contentsOf: tokens)
            accumulatedTokenTimings.append(contentsOf: timings)
            if let (tokenId, frame) = thenEmit {
                accumulatedTokenIds.append(tokenId)
                appendTokenTiming(
                    tokenId, frameInChunk: frame - absoluteFrameBase, tokenizer: tokenizer, observeBias: false)
            }
        case .rewound(let idsCount, let timingsCount, let frame):
            accumulatedTokenIds.removeSubrange(idsCount...)
            accumulatedTokenTimings.removeSubrange(timingsCount...)
            if let current = session.chunks.last, frame < current.startFrame {
                session.decoder.noteCrossChunkRewind()
            }
        }
    }

    /// Timings for a word re-spelled as `tokens`: the replaced word's time
    /// span (through absolute `wordEndFrame` when a piece of it has no timing
    /// yet), spread evenly over the new tokens, so word timings stay where
    /// the audio was.
    nonisolated internal static func respelledTimings(
        _ tokens: [Int], replacing replaced: ArraySlice<TokenTiming>, wordEndFrame: Int?,
        piece: (Int) -> String
    ) -> [TokenTiming] {
        let frameSeconds = ASRConstants.secondsPerEncoderFrame
        let spanEnd = max(replaced.last?.endTime ?? 0, wordEndFrame.map { Double($0 + 1) * frameSeconds } ?? 0)
        let spanStart = replaced.first?.startTime ?? max(0, spanEnd - frameSeconds)
        let step = max(spanEnd - spanStart, frameSeconds) / Double(max(tokens.count, 1))
        return tokens.enumerated().map { index, tokenId in
            let start = spanStart + Double(index) * step
            return TokenTiming(
                token: piece(tokenId), tokenId: tokenId, startTime: start, endTime: start + step, confidence: 1.0)
        }
    }

    // MARK: - Step providers

    private var biasSmartSpeculativeEnabled: Bool {
        if let v = ProcessInfo.processInfo.environment["FLUIDAUDIO_ENABLE_SMART_SPECULATIVE"] {
            let lowered = v.lowercased()
            return !(lowered == "0" || lowered == "false" || lowered == "no")
        }
        return true
    }

    /// Logits + next state for the decoder's current input. Mirrors the
    /// unbiased paths: the speculative fast path (batched joint over K frames
    /// for the first emission at a frame, the fused drain model after it)
    /// when its assets are loaded, else the per-frame fused decoder.
    private func biasedStep(
        _ session: NemotronBiasDecodeSession
    ) async throws -> (NemotronStepLogits, NemotronDecoderState) {
        let decoder = session.decoder
        let frame = decoder.frame
        guard let chunk = session.chunks.last(where: { $0.contains(frame) }) else {
            throw ASRError.processingFailed("Biased decode: encoder frame \(frame) is no longer retained")
        }
        let local = frame - chunk.startFrame
        if decoder.symbol == 0, biasSmartSpeculativeEnabled, let jointBatched = jointNoEncProjBatched,
            let predictor = self.decoder, let proj = chunk.encoderProj
        {
            return try await batchedStep(
                session, chunk: chunk, local: local, encoderProj: proj, predictor: predictor,
                jointBatched: jointBatched)
        }
        return try await singleStep(chunk: chunk, local: local, token: decoder.token, state: decoder.state)
    }

    /// One fused decoder+joint call at one frame: B3 (pre-projected encoder)
    /// → B1 → separate decoder + joint. The fused-argmax model is never
    /// used here — it exposes no logits.
    private func singleStep(
        chunk: NemotronRetainedChunk, local: Int, token: Int, state: NemotronDecoderState
    ) async throws -> (NemotronStepLogits, NemotronDecoderState) {
        let tokenInput: MLMultiArray
        if let buf = tokenInputBuf {
            tokenInput = buf
        } else {
            tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        }
        tokenInput[0] = NSNumber(value: Int32(token))
        let tokenLen: MLMultiArray
        if let buf = tokenLenBuf {
            tokenLen = buf
        } else {
            tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
            tokenLen[0] = 1
        }

        if let djne = decoderJointNoEncProj, let proj = chunk.encoderProj {
            let projStep: MLMultiArray
            if let buf = encoderProjStepBuf {
                fillEncoderProjStep(into: buf, from: proj, timeIndex: local)
                projStep = buf
            } else {
                projStep = try extractEncoderProjStep(from: proj, timeIndex: local)
            }
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: state.h),
                "c_in": MLFeatureValue(multiArray: state.c),
                "encoder_proj": MLFeatureValue(multiArray: projStep),
            ])
            let output: MLFeatureProvider
            if let opts = decoderJointNoEncProjPredictionOptions {
                output = try await djne.prediction(from: input, options: opts)
            } else {
                output = try await djne.prediction(from: input)
            }
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                let h = output.featureValue(for: "h_out")?.multiArrayValue,
                let c = output.featureValue(for: "c_out")?.multiArrayValue
            else { throw ASRError.processingFailed("B3+B1 fused decoder_joint_noencproj failed") }
            return (contiguousLogits(logits), NemotronDecoderState(h: h, c: c))
        }

        let encStep: MLMultiArray
        if let buf = encoderStepBuf {
            fillEncoderStep(into: buf, from: chunk.encoded, timeIndex: local)
            encStep = buf
        } else {
            encStep = try extractEncoderStep(from: chunk.encoded, timeIndex: local)
        }
        if let dj = decoderJoint {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: state.h),
                "c_in": MLFeatureValue(multiArray: state.c),
                "encoder": MLFeatureValue(multiArray: encStep),
            ])
            let output: MLFeatureProvider
            if let opts = decoderJointPredictionOptions {
                output = try await dj.prediction(from: input, options: opts)
            } else {
                output = try await dj.prediction(from: input)
            }
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                let h = output.featureValue(for: "h_out")?.multiArrayValue,
                let c = output.featureValue(for: "c_out")?.multiArrayValue
            else { throw ASRError.processingFailed("Fused decoder_joint failed") }
            return (contiguousLogits(logits), NemotronDecoderState(h: h, c: c))
        }
        if let predictor = self.decoder, let joint = self.joint {
            let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
                "token": MLFeatureValue(multiArray: tokenInput),
                "token_length": MLFeatureValue(multiArray: tokenLen),
                "h_in": MLFeatureValue(multiArray: state.h),
                "c_in": MLFeatureValue(multiArray: state.c),
            ])
            let decoderOutput: MLFeatureProvider
            if let opts = decoderPredictionOptions {
                decoderOutput = try await predictor.prediction(from: decoderInput, options: opts)
            } else {
                decoderOutput = try await predictor.prediction(from: decoderInput)
            }
            guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
                let h = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
                let c = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
            else { throw ASRError.processingFailed("Decoder failed") }
            let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                "encoder": MLFeatureValue(multiArray: encStep),
                "decoder": MLFeatureValue(multiArray: try sliceDecoderOutput(decoderOut)),
            ])
            let jointOutput: MLFeatureProvider
            if let opts = jointPredictionOptions {
                jointOutput = try await joint.prediction(from: jointInput, options: opts)
            } else {
                jointOutput = try await joint.prediction(from: jointInput)
            }
            guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
                throw ASRError.processingFailed("Joint failed")
            }
            return (contiguousLogits(logits), NemotronDecoderState(h: h, c: c))
        }
        throw ASRError.processingFailed(
            "Custom vocabulary needs a logits-producing decoder (decoder_joint_noencproj, decoder_joint, "
                + "or decoder + joint)")
    }

    /// Speculative fast path, first emission at a frame: one decoder call and
    /// one batched joint over the next K frames of the chunk, shared by every
    /// frame until an emission changes the decoder input (blank emissions do
    /// not advance RNN-T state, so those K logits are exact).
    private func batchedStep(
        _ session: NemotronBiasDecodeSession,
        chunk: NemotronRetainedChunk,
        local: Int,
        encoderProj: MLMultiArray,
        predictor: MLModel,
        jointBatched: MLModel
    ) async throws -> (NemotronStepLogits, NemotronDecoderState) {
        let frame = chunk.startFrame + local
        let token = session.decoder.token
        let state = session.decoder.state
        if let cached = session.batch, cached.h === state.h, cached.token == token,
            frame >= cached.firstFrame, frame < cached.firstFrame + cached.count
        {
            return (batchedLogits(cached.logits, row: frame - cached.firstFrame), cached.state)
        }

        let K = jointNoEncProjBatchedK
        let kActual = min(K, chunk.frameCount - local)
        let tokenInput: MLMultiArray
        if let buf = tokenInputBuf {
            tokenInput = buf
        } else {
            tokenInput = try MLMultiArray(shape: [1, 1], dataType: .int32)
        }
        tokenInput[0] = NSNumber(value: Int32(token))
        let tokenLen: MLMultiArray
        if let buf = tokenLenBuf {
            tokenLen = buf
        } else {
            tokenLen = try MLMultiArray(shape: [1], dataType: .int32)
            tokenLen[0] = 1
        }
        let decoderInput = try MLDictionaryFeatureProvider(dictionary: [
            "token": MLFeatureValue(multiArray: tokenInput),
            "token_length": MLFeatureValue(multiArray: tokenLen),
            "h_in": MLFeatureValue(multiArray: state.h),
            "c_in": MLFeatureValue(multiArray: state.c),
        ])
        let decoderOutput: MLFeatureProvider
        if let opts = decoderPredictionOptions {
            decoderOutput = try await predictor.prediction(from: decoderInput, options: opts)
        } else {
            decoderOutput = try await predictor.prediction(from: decoderInput)
        }
        guard let decoderOut = decoderOutput.featureValue(for: "decoder_out")?.multiArrayValue,
            let h = decoderOutput.featureValue(for: "h_out")?.multiArrayValue,
            let c = decoderOutput.featureValue(for: "c_out")?.multiArrayValue
        else { throw ASRError.processingFailed("Speculative decoder failed") }

        if encProjBatchReusable == nil || encProjBatchReusable?.shape[1].intValue != K {
            encProjBatchReusable = try MLMultiArray(shape: [1, NSNumber(value: K), 640], dataType: .float32)
        }
        let batchBuf = encProjBatchReusable!
        let projDim = encoderProj.shape[2].intValue
        let src = encoderProj.dataPointer.bindMemory(to: Float.self, capacity: encoderProj.count)
        let dst = batchBuf.dataPointer.bindMemory(to: Float.self, capacity: batchBuf.count)
        for k in 0..<K {
            let base = k * projDim
            if k < kActual {
                let srcBase = (local + k) * projDim  // contiguous copy: [1, T, D]
                for d in 0..<projDim { dst[base + d] = src[srcBase + d] }
            } else {
                for d in 0..<projDim { dst[base + d] = 0 }
            }
        }
        let jointInput = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_proj": MLFeatureValue(multiArray: batchBuf),
            "decoder": MLFeatureValue(multiArray: try sliceDecoderOutput(decoderOut)),
        ])
        let jointOutput: MLFeatureProvider
        if let opts = jointNoEncProjBatchedPredictionOptions {
            jointOutput = try await jointBatched.prediction(from: jointInput, options: opts)
        } else {
            jointOutput = try await jointBatched.prediction(from: jointInput)
        }
        guard let logits = jointOutput.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Speculative joint_noencproj_batched failed")
        }
        specWindowsTotal &+= 1
        let next = NemotronDecoderState(h: h, c: c)
        session.batch = (state.h, token, frame, kActual, logits, next)
        return (batchedLogits(logits, row: 0), next)
    }

    /// Contiguous Float32 logits (every fused/joint single-step output).
    private func contiguousLogits(_ logits: MLMultiArray) -> NemotronStepLogits {
        let count = logits.count
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: count)
        return NemotronStepLogits(count: count, argmax: findMaxIndex(logits)) { ptr[$0] }
    }

    /// One row of batched-joint logits [1, K, 1, V] (Float16 or Float32,
    /// possibly strided).
    private func batchedLogits(_ logits: MLMultiArray, row: Int) -> NemotronStepLogits {
        let vocab = logits.shape[3].intValue
        let base = row * logits.strides[1].intValue
        let stride = logits.strides[3].intValue
        if logits.dataType == .float16 {
            let ptr = logits.dataPointer.bindMemory(to: UInt16.self, capacity: logits.count)
            let read: (Int) -> Float = { nemotronHalfBitsToFloat(ptr[base + $0 * stride]) }
            var best = 0
            var bestValue = -Float.greatestFiniteMagnitude
            for v in 0..<vocab {
                let value = read(v)
                if value > bestValue {
                    bestValue = value
                    best = v
                }
            }
            return NemotronStepLogits(count: vocab, argmax: best, logit: read)
        }
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        var best = 0
        if stride == 1 {
            var maxValue: Float = 0
            var maxIndex: vDSP_Length = 0
            vDSP_maxvi(ptr.advanced(by: base), 1, &maxValue, &maxIndex, vDSP_Length(vocab))
            best = Int(maxIndex)
        } else {
            var bestValue = -Float.greatestFiniteMagnitude
            for v in 0..<vocab where ptr[base + v * stride] > bestValue {
                bestValue = ptr[base + v * stride]
                best = v
            }
        }
        return NemotronStepLogits(count: vocab, argmax: best) { ptr[base + $0 * stride] }
    }

    /// Deep copy of a rank-3 Float32/Float16 tensor into a fresh contiguous
    /// Float32 array of the same shape.
    nonisolated internal static func contiguousFloat32Copy(_ source: MLMultiArray) throws -> MLMultiArray {
        let shape = source.shape.map { $0.intValue }
        precondition(shape.count == 3, "expected a rank-3 encoder tensor")
        let copy = try MLMultiArray(shape: source.shape, dataType: .float32)
        let dst = copy.dataPointer.bindMemory(to: Float.self, capacity: copy.count)
        let s = source.strides.map { $0.intValue }
        var index = 0
        if source.dataType == .float16 {
            let src = source.dataPointer.bindMemory(to: UInt16.self, capacity: source.count)
            for a in 0..<shape[0] {
                for b in 0..<shape[1] {
                    for c in 0..<shape[2] {
                        dst[index] = nemotronHalfBitsToFloat(src[a * s[0] + b * s[1] + c * s[2]])
                        index += 1
                    }
                }
            }
        } else {
            let src = source.dataPointer.bindMemory(to: Float.self, capacity: source.count)
            for a in 0..<shape[0] {
                for b in 0..<shape[1] {
                    for c in 0..<shape[2] {
                        dst[index] = src[a * s[0] + b * s[1] + c * s[2]]
                        index += 1
                    }
                }
            }
        }
        return copy
    }
}
