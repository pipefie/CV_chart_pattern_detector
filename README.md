## WeaK labeling on the rendered set 

We use the data we already have (the parquet OHLCV) to detect patterns of the price series (not the pixels) then convert the detected pattern’s geometry to pixel coordinates using each image’s sidecar metadata. That gives us labels without manual annotation 


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


## Glosary

ATR = average true range