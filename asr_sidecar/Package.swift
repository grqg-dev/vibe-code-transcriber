// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "asr-sidecar",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .executable(name: "asr-sidecar", targets: ["ASRSidecar"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            exact: "0.15.1"
        ),
    ],
    targets: [
        .executableTarget(
            name: "ASRSidecar",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources"
        ),
    ]
)
