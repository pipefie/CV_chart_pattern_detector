## How to run the standardization script for any type of image

### Screeenshots

uv run python src/standardize/standardize_images.py \
  --inp data/images/real/screenshots \
  --out data/images/standardized/screenshots

### Phone Photos

uv run python src/standardize/standardize_images.py \
  --inp data/images/real/photos \
  --out data/images/standardized/photos


## Rendered Images

uv run python src/standardize/standardize_images.py \
  --inp data/images/rendered/val \
  --out data/images/standardized/rendered_val
