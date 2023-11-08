# Changelog

All notable changes to Pixano will be documented in this file.



## [Unreleased]



## [0.2.0] - 2023-10-26

### Changed
- **Breaking:** Update models to the new **PixanoTypes** and **lancedb storage format** of Pixano 0.4.0



## [0.1.6] - 2023-07-10

### Added
- Create documentation website

### Fixed
- Update CHANGELOG format



## [0.1.5] - 2023-07-07

### Fixed
- Fix README logo and links



## [0.1.4] - 2023-07-07

### Fixed
- Fix export to ONNX for SAM



## [0.1.3] - 2023-07-07

### Changed
- **Breaking:** Update models to the new **InferenceModel class** and **Image type** of Pixano 0.3.0


## [0.1.2] - 2023-06-12

### Added
- Add release notes in new RELEASE.md file

### Fixed
- Update project description in README
- Update Python version requirement and License classifier in pyproject.toml



## [0.1.1] - 2023-06-08

### Fixed
- Update COCO 80 labels to COCO 91 labels function name for consistency with Pixano API
- Update README badges



## [0.1.0] - 2023-06-02

### Changed
- **Breaking:** Merge inference generation and embedding precomputing into a **single InferenceModel class**
- Improve README file

### Fixed
- Order all image transforms args to height first and width second for consistency
- Fix COCO labels mismatch between YOLO predictions on 80 labels and ground truth on 91 labels
- Convert TensorFlow models bounding box coordinates from yxyx to xyxy before xywh conversion



## [0.0.1] - 2023-05-11

### Added
- Create first public release



[Unreleased]: https://github.com/pixano/pixano/compare/main...develop
[0.2.0]: https://github.com/pixano/pixano-inference/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/pixano/pixano-inference/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/pixano/pixano-inference/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/pixano/pixano-inference/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/pixano/pixano-inference/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/pixano/pixano-inference/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/pixano/pixano-inference/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pixano/pixano-inference/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/pixano/pixano-inference/releases/tag/v0.0.1
