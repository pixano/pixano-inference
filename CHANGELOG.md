# Changelog

All notable changes to Pixano will be documented in this file.

## [Unreleased]

## [0.3.1] - 2024-03-18

### Added

- Add **new GroundingDINO model** for semantic segmentation with text prompts (pixano/pixano-inference#6)

### Changed

- Update README badges with PyPI release

### Fixed

- Remove top-level imports for GitHub models to prevent import errors (pixano/pixano-inference#6)
- Fix preannotation with SAM and MobileSAM (pixano/pixano-inference#6)
- Add type hints for Image PixanoType (pixano/pixano-inference#6)
- Update Pixano requirement from 0.5.0 beta to 0.5.0 stable

## [0.3.0] - 2024-02-29

### Added

- Publish **Pixano Inference on PyPI**
- Add **new MobileSAM model** as a lighter alternative to SAM (pixano/pixano-inference#2)
- Add GitHub actions to format and lint code
- Add GitHub action to publish docs and PyPI package (pixano/pixano-inference#9)
- Add issue and pull request templates on GitHub repository
- Add CONTRIBUTING.md for installation information and contribution guidelines

### Changed

- **Breaking:** Remove SAM and MobileSAM dependencies to allow publishing to PyPI (pixano/pixano-inference#14)
- **Breaking:** Update to Pixano 0.5.0
- **Breaking:** Update InferenceModel `id` attribute to `model_id` to stop redefining built-in `id`
- **Breaking:** Update submodule names to `pytorch`, `tensorflow`, and `github`
- Update README with a small header description listing main features and more detailed installation instructions
- Generate API reference on documentation website automatically
- Add cross-references to Pixano, TensorFlow, and Hugging Face Transformers in the API reference
- Update documentation deployment (pixano/pixano-inference#9)

### Fixed

- Fix links to Pixano notebooks
- Fix internal cross-references in the API reference of the documentation website
- Update deprecated GitHub actions (pixano/pixano-inference#11)
- Prevent CUDA installation in lint GitHub action action
- Fix ignored members in linting CI configuration
- Specify black version in formatting CI configuration
- Fix GitHub version and documentation links in README

## [0.2.1] - 2023-11-13

### Added

- Add CLIP model for **semantic search** on images

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

[Unreleased]: https://github.com/pixano/pixano-inference/compare/main...develop
[0.3.1]: https://github.com/pixano/pixano-inference/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/pixano/pixano-inference/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/pixano/pixano-inference/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/pixano/pixano-inference/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/pixano/pixano-inference/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/pixano/pixano-inference/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/pixano/pixano-inference/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/pixano/pixano-inference/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/pixano/pixano-inference/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/pixano/pixano-inference/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pixano/pixano-inference/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/pixano/pixano-inference/releases/tag/v0.0.1
