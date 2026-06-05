#if os(macOS)
@preconcurrency import AVFoundation
import Darwin
import FluidAudio
import Foundation

// Long-lived ASR subprocess for vibe-code-transcriber.
// Protocol: JSON lines on stdin (requests) and stdout (responses).
// Crash traces go to stderr only so stdout stays machine-parseable.

private struct InMessage: Decodable {
    let type: String
    let id: Int?
    let path: String?
}

private struct ReadyOut: Encodable {
    let type = "ready"
}

private struct ResultOut: Encodable {
    let type = "result"
    let id: Int
    let text: String
    let audio_s: Double
    let processing_s: Double
    let ok = true
}

private struct ErrorOut: Encodable {
    let type = "error"
    let id: Int?
    let message: String
    let ok = false
}

private func emit<T: Encodable>(_ value: T, stdout: FileHandle) {
    let data = try! JSONEncoder().encode(value)
    guard var line = String(data: data, encoding: .utf8) else { return }
    line.append("\n")
    stdout.write(line.data(using: .utf8)!)
    try? stdout.synchronize()
}

private func emitError(id: Int?, message: String, stdout: FileHandle) {
    emit(ErrorOut(id: id, message: message), stdout: stdout)
}

private func parseModelVersion(_ arg: String) -> AsrModelVersion? {
    switch arg.lowercased() {
    case "v2": return .v2
    case "v3": return .v3
    default: return nil
    }
}

private func audioDurationSeconds(at url: URL) -> Double {
    if let file = try? AVAudioFile(forReading: url) {
        let rate = file.processingFormat.sampleRate
        guard rate > 0 else { return 0 }
        return Double(file.length) / rate
    }
    return 0
}

@main
struct ASRSidecar {
    static func main() async {
        let stdout = FileHandle.standardOutput

        let args = CommandLine.arguments
        var modelVersion: AsrModelVersion = .v2
        var i = 1
        while i < args.count {
            if args[i] == "--model-version", i + 1 < args.count {
                if let v = parseModelVersion(args[i + 1]) {
                    modelVersion = v
                } else {
                    fputs("Unknown --model-version \(args[i + 1]) (use v2 or v3)\n", Darwin.stderr)
                    exit(2)
                }
                i += 2
                continue
            }
            i += 1
        }

        let asrManager: AsrManager
        do {
            fputs("Loading FluidAudio Parakeet (\(modelVersion))...\n", Darwin.stderr)
            let models = try await AsrModels.downloadAndLoad(version: modelVersion)
            let tdtConfig = TdtConfig(blankId: modelVersion.blankId)
            let asrConfig = ASRConfig(
                tdtConfig: tdtConfig,
                encoderHiddenSize: modelVersion.encoderHiddenSize
            )
            asrManager = AsrManager(config: asrConfig)
            try await asrManager.loadModels(models)

            // Warmup: 1s silence at 16 kHz (matches Python warmup_model).
            let silence = [Float](repeating: 0, count: 16_000)
            var warmupState = TdtDecoderState.make(
                decoderLayers: await asrManager.decoderLayerCount
            )
            _ = try await asrManager.transcribe(
                silence, decoderState: &warmupState
            )
        } catch {
            fputs("ASR init failed: \(error)\n", Darwin.stderr)
            exit(1)
        }

        emit(ReadyOut(), stdout: stdout)

        while let line = readLine() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }

            let msg: InMessage
            do {
                msg = try JSONDecoder().decode(InMessage.self, from: Data(trimmed.utf8))
            } catch {
                emitError(id: nil, message: "invalid JSON: \(error)", stdout: stdout)
                continue
            }

            switch msg.type {
            case "quit":
                return
            case "transcribe":
                guard let id = msg.id, let path = msg.path else {
                    emitError(id: msg.id, message: "transcribe requires id and path", stdout: stdout)
                    continue
                }
                let url = URL(fileURLWithPath: path)
                guard FileManager.default.fileExists(atPath: path) else {
                    emitError(id: id, message: "file not found: \(path)", stdout: stdout)
                    continue
                }
                let audioS = audioDurationSeconds(at: url)
                do {
                    var decoderState = TdtDecoderState.make(
                        decoderLayers: await asrManager.decoderLayerCount
                    )
                    let start = Date()
                    let result = try await asrManager.transcribe(
                        url, decoderState: &decoderState
                    )
                    let processingS = Date().timeIntervalSince(start)
                    emit(
                        ResultOut(
                            id: id,
                            text: result.text.trimmingCharacters(in: .whitespacesAndNewlines),
                            audio_s: audioS,
                            processing_s: processingS
                        ),
                        stdout: stdout
                    )
                } catch {
                    emitError(id: id, message: String(describing: error), stdout: stdout)
                }
            default:
                // Forward-compatible: ignore unknown types.
                break
            }
        }
    }
}
#endif
