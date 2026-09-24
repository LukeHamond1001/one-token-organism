// THE PARENT'S VOICE ENGINE (docs/SIM_DESIGN.md 4.4; package P1): a small server on macOS speech (AVSpeechSynthesizer.write),
// one warm synthesizer, run off the tick loop at nice 10 by body/sim/voice/synth.py, which builds this file with swiftc.
//
// Protocol: one JSON object a line on stdin, one JSON object a line on stdout.
//   {"op":"synth","id":N,"voice":ID,"rate":R,"pitch":P,"volume":V,"text":S,"ssml":false,"out":PATH}
//       writes the utterance's samples to PATH as raw float32 mono at the engine's own rate, and answers
//       {"id":N,"ok":true,"sr":22050,"n":FRAMES,"bpf":4,"marks":[[frame,loc,len],...],"secs":WALL}
//       (marks: the engine's word markers, each its first frame and its range in the text as given; the byte offset the
//       engine reports is divided by the buffer's bytes per frame here, so a mark is a sample index)
//   {"op":"info","voice":ID}   -> {"ok":true,"name":..,"quality":..,"language":..,"identifier":..,"os":..}
//   {"op":"voices"}            -> {"ok":true,"voices":[[identifier,name,language,quality],...]}
//   {"op":"quit"}
// An error answers {"id":N,"ok":false,"error":TEXT}. Nothing here draws a random number; the engine's own output is measured to be
// identical run to run (body/tests/test_sim_voice.py), and the cache in synth.py keeps every clip's digest besides.
import AVFoundation
import Foundation

setvbuf(stdout, nil, _IOLBF, 0)
let synth = AVSpeechSynthesizer()

func reply(_ obj: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: d, encoding: .utf8) {
        print(s)
    } else {
        print("{\"ok\":false,\"error\":\"unencodable reply\"}")
    }
    fflush(stdout)
}

func quality(_ v: AVSpeechSynthesisVoice) -> String {
    switch v.quality {
    case .premium: return "premium"
    case .enhanced: return "enhanced"
    default: return "default"
    }
}

func synthesize(_ req: [String: Any]) -> [String: Any] {
    let id = req["id"] ?? 0
    guard let vid = req["voice"] as? String, let voice = AVSpeechSynthesisVoice(identifier: vid) else {
        return ["id": id, "ok": false, "error": "no such voice"]
    }
    guard let text = req["text"] as? String, let out = req["out"] as? String else {
        return ["id": id, "ok": false, "error": "text and out are required"]
    }
    let isSSML = (req["ssml"] as? Bool) ?? false
    var u: AVSpeechUtterance
    if isSSML {
        guard let s = AVSpeechUtterance(ssmlRepresentation: text) else {
            return ["id": id, "ok": false, "error": "the SSML was refused"]
        }
        u = s
    } else {
        u = AVSpeechUtterance(string: text)
    }
    u.voice = voice
    u.rate = Float((req["rate"] as? Double) ?? Double(AVSpeechUtteranceDefaultSpeechRate))
    u.pitchMultiplier = Float((req["pitch"] as? Double) ?? 1.0)
    u.volume = Float((req["volume"] as? Double) ?? 1.0)
    u.preUtteranceDelay = 0
    u.postUtteranceDelay = 0
    var samples: [Float] = []
    var sr = 0.0
    var bpf: UInt32 = 0
    var marks: [[Int]] = []
    var done = false
    var failed: String? = nil
    let t0 = Date()
    synth.write(u, toBufferCallback: { buf in
        guard let pcm = buf as? AVAudioPCMBuffer else { return }
        if pcm.frameLength == 0 { done = true; return }
        sr = pcm.format.sampleRate
        bpf = pcm.format.streamDescription.pointee.mBytesPerFrame
        if pcm.format.channelCount != 1 { failed = "not mono" }
        if let ch = pcm.floatChannelData {
            samples.append(contentsOf: UnsafeBufferPointer(start: ch[0], count: Int(pcm.frameLength)))
        } else if let ch = pcm.int16ChannelData {
            for k in 0..<Int(pcm.frameLength) { samples.append(Float(ch[0][k]) / 32768) }
        } else {
            failed = "an unknown sample format"
        }
    }, toMarkerCallback: { ms in
        for m in ms where m.mark == .word {
            let per = Int(max(bpf, 1))
            marks.append([m.byteSampleOffset / per, m.textRange.location, m.textRange.length])
        }
    })
    while !done && Date().timeIntervalSince(t0) < 30 {
        RunLoop.current.run(until: Date().addingTimeInterval(0.001))
    }
    if !done { return ["id": id, "ok": false, "error": "the engine did not finish in 30 s"] }
    if let f = failed { return ["id": id, "ok": false, "error": f] }
    let data = samples.withUnsafeBufferPointer { Data(buffer: $0) }
    do { try data.write(to: URL(fileURLWithPath: out)) } catch {
        return ["id": id, "ok": false, "error": "could not write \(out)"]
    }
    return ["id": id, "ok": true, "sr": sr, "n": samples.count, "bpf": Int(bpf), "marks": marks,
            "secs": Date().timeIntervalSince(t0)]
}

while let line = readLine(strippingNewline: true) {
    guard let d = line.data(using: .utf8), let obj = try? JSONSerialization.jsonObject(with: d), let req = obj as? [String: Any]
    else { reply(["ok": false, "error": "not a JSON object"]); continue }
    let op = (req["op"] as? String) ?? "synth"
    if op == "quit" { break }
    if op == "info" {
        guard let vid = req["voice"] as? String, let v = AVSpeechSynthesisVoice(identifier: vid) else {
            reply(["ok": false, "error": "no such voice"]); continue
        }
        reply(["ok": true, "name": v.name, "quality": quality(v), "language": v.language, "identifier": v.identifier,
               "os": ProcessInfo.processInfo.operatingSystemVersionString])
        continue
    }
    if op == "voices" {
        let vs = AVSpeechSynthesisVoice.speechVoices().map { [$0.identifier, $0.name, $0.language, quality($0)] }
        reply(["ok": true, "voices": vs])
        continue
    }
    reply(synthesize(req))
}
