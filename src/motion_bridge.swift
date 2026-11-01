import CoreMotion
import Foundation

// AirPods head motion, bridged to Rust for the training-data capture (Labs ›
// AirPods training data). Motion updates run only between
// `hex_motion_start` and `hex_motion_stop`, which follow a dictation's
// shortcut press and release. Nothing is written to disk here.

/// How many samples one session may accumulate: five minutes at 25 Hz. A
/// capture window is read from this buffer at the release; anything older is
/// dropped, so a stuck capture cannot grow without bound.
private let capacity = 25 * 60 * 5

private struct BridgeMotionSample: Codable {
    /// `CMDeviceMotion.timestamp` in nanoseconds since boot, the same clock
    /// as Hex's `CaptureInstant` (`mach_absolute_time`).
    let t: Double
    /// User acceleration in g, without gravity.
    let ax: Double
    let ay: Double
    let az: Double
    /// Rotation rate in radians per second.
    let rx: Double
    let ry: Double
    let rz: Double
    /// Attitude in radians.
    let pitch: Double
    let roll: Double
    let yaw: Double
    /// Gravity in g.
    let gx: Double
    let gy: Double
    let gz: Double
    /// Attitude quaternion.
    let qw: Double
    let qx: Double
    let qy: Double
    let qz: Double
}

/// The one manager: CoreMotion keeps device motion attached to the instance
/// that started it, and a second one would deliver no updates.
private let manager = CMHeadphoneMotionManager()
private let queue: OperationQueue = {
    let queue = OperationQueue()
    queue.name = "hex.motion"
    queue.maxConcurrentOperationCount = 1
    queue.qualityOfService = .utility
    return queue
}()

/// Serializes the sample buffer, connection state, and session counter:
/// updates arrive on `queue`, reads come from whichever thread asks.
private let lock = NSLock()
private var samples: [BridgeMotionSample] = []
private var connected = false
private var sessionCount: UInt64 = 0
private var currentSession: UInt64 = 0
private var running = false
private var lastError: String?

private final class Delegate: NSObject, CMHeadphoneMotionManagerDelegate {
    func headphoneMotionManagerDidConnect(_ manager: CMHeadphoneMotionManager) {
        lock.lock()
        defer { lock.unlock() }
        // A reconnect (the AirPods were re-seated) starts a new session: the
        // trainer splits on it, because the signal changes with placement.
        if !connected {
            connected = true
            sessionCount &+= 1
            currentSession = sessionCount
        }
    }

    func headphoneMotionManagerDidDisconnect(_ manager: CMHeadphoneMotionManager) {
        lock.lock()
        defer { lock.unlock() }
        connected = false
    }
}

private let delegate = Delegate()

private func record(_ motion: CMDeviceMotion) {
    let sample = BridgeMotionSample(
        t: motion.timestamp * 1_000_000_000,
        ax: motion.userAcceleration.x,
        ay: motion.userAcceleration.y,
        az: motion.userAcceleration.z,
        rx: motion.rotationRate.x,
        ry: motion.rotationRate.y,
        rz: motion.rotationRate.z,
        pitch: motion.attitude.pitch,
        roll: motion.attitude.roll,
        yaw: motion.attitude.yaw,
        gx: motion.gravity.x,
        gy: motion.gravity.y,
        gz: motion.gravity.z,
        qw: motion.attitude.quaternion.w,
        qx: motion.attitude.quaternion.x,
        qy: motion.attitude.quaternion.y,
        qz: motion.attitude.quaternion.z
    )
    lock.lock()
    defer { lock.unlock() }
    // The first sample after a start also proves the connection, for a pair
    // that was already worn when updates began and sent no connect event.
    if !connected {
        connected = true
        sessionCount &+= 1
        currentSession = sessionCount
    }
    samples.append(sample)
    if samples.count > capacity {
        samples.removeFirst(samples.count - capacity)
    }
}

/// CoreMotion's shared status: 0 not determined, 1 restricted, 2 denied,
/// 3 authorized.
private func statusCode(_ status: CMAuthorizationStatus) -> Int32 {
    switch status {
    case .notDetermined: return 0
    case .restricted: return 1
    case .denied: return 2
    case .authorized: return 3
    @unknown default: return 2
    }
}

@_cdecl("hex_motion_authorization")
public func hexMotionAuthorization() -> Int32 {
    statusCode(CMHeadphoneMotionManager.authorizationStatus())
}

/// Whether this Mac and its headphones can stream motion at all.
@_cdecl("hex_motion_available")
public func hexMotionAvailable() -> Bool {
    manager.isDeviceMotionAvailable
}

/// Starts motion updates, which on first use shows CoreMotion's permission
/// prompt. Returns 0 when running, 1 when the hardware cannot, 2 when access
/// is denied, 3 when CoreMotion reported an error (see `hex_motion_error`).
@_cdecl("hex_motion_start")
public func hexMotionStart() -> Int32 {
    switch hexMotionAuthorization() {
    case 1, 2:
        return 2
    default:
        break
    }
    guard manager.isDeviceMotionAvailable else { return 1 }
    lock.lock()
    if running {
        lock.unlock()
        return 0
    }
    running = true
    lastError = nil
    samples.removeAll(keepingCapacity: true)
    lock.unlock()
    manager.delegate = delegate
    manager.startDeviceMotionUpdates(to: queue) { motion, error in
        if let error {
            let message = error.localizedDescription
            lock.lock()
            lastError = message
            lock.unlock()
            return
        }
        if let motion {
            record(motion)
        }
    }
    return 0
}

@_cdecl("hex_motion_stop")
public func hexMotionStop() {
    lock.lock()
    let wasRunning = running
    running = false
    lastError = nil
    samples.removeAll(keepingCapacity: true)
    lock.unlock()
    if wasRunning {
        manager.stopDeviceMotionUpdates()
    }
}

@_cdecl("hex_motion_running")
public func hexMotionRunning() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return running
}

@_cdecl("hex_motion_connected")
public func hexMotionConnected() -> Bool {
    lock.lock()
    defer { lock.unlock() }
    return connected
}

/// Increases each time the headphones connect (or begin delivering motion
/// after a reconnect), never repeating within a run.
@_cdecl("hex_motion_session")
public func hexMotionSession() -> UInt64 {
    lock.lock()
    defer { lock.unlock() }
    return currentSession
}

/// CoreMotion's last error from an update, if any (owned; free it with
/// `hex_motion_free_string`).
@_cdecl("hex_motion_error")
public func hexMotionError() -> UnsafeMutablePointer<CChar>? {
    lock.lock()
    defer { lock.unlock() }
    return lastError.map { strdup($0) }
}

/// The samples with `from <= t <= to` (nanoseconds), oldest first, as JSON.
/// Reading does not remove them; `hex_motion_stop` drops the buffer.
@_cdecl("hex_motion_samples")
public func hexMotionSamples(_ from: Double, _ to: Double) -> UnsafeMutablePointer<CChar>? {
    lock.lock()
    let selected = samples.filter { $0.t >= from && $0.t <= to }
    lock.unlock()
    guard let data = try? JSONEncoder().encode(selected) else { return nil }
    return strdup(String(decoding: data, as: UTF8.self))
}

/// The newest buffered sample's time in nanoseconds, or 0 when the buffer is
/// empty; the HUD's motion indicator polls this to tell a live stream from a
/// connected pair that sends nothing.
@_cdecl("hex_motion_newest_sample")
public func hexMotionNewestSample() -> Double {
    lock.lock()
    defer { lock.unlock() }
    return samples.last?.t ?? 0
}

@_cdecl("hex_motion_free_string")
public func hexMotionFreeString(_ string: UnsafeMutablePointer<CChar>?) {
    free(string)
}
