# Release notes


# Pixano Inference v0.1.2

### Added
- Add release notes in new RELEASE.md file

### Fixed
- Update project description in README
- Update Python version requirement and License classifier in pyproject.toml


## Pixano Inference v0.1.1

### Fixed
- Update COCO 80 labels to COCO 91 labels function name for consistency with Pixano API
- Update README badges


## Pixano Inference v0.1.0

### Changed
- Merge inference generation and embedding precomputing into a **single inference model class**
- Improve README file

### Fixed
- Order all image transforms args to height first and width second for consistency
- Fix COCO labels mismatch between YOLO predictions on 80 labels and ground truth on 91 labels
- Convert TensorFlow models bounding box coordinates from yxyx to xyxy before xywh conversion


## Pixano Inference v0.0.1

### Added
- Create first public release