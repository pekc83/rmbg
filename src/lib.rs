use anyhow::anyhow;
use fast_image_resize as fr;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, Rgba, RgbaImage};
use ndarray::{s, Array3, ArrayView, Axis, Dim};
use std::num::NonZeroU32;
use std::path::Path;

const ML_MODEL_IMAGE_WIDTH: u32 = 1024;
const ML_MODEL_IMAGE_HEIGHT: u32 = 1024;
const ML_MODEL_INPUT_NAME: &str = "input";
const ML_MODEL_OUTPUT_NAME: &str = "output";

pub struct Rmbg {
    model: ort::Session,
}

impl Rmbg {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, ort::Error> {
        let model = ort::Session::builder()?.commit_from_file(model_path)?;
        Ok(Rmbg { model })
    }

    pub fn remove_background(&self, original_img: &DynamicImage) -> anyhow::Result<DynamicImage> {
        let img = preprocess_image(original_img)?;

        let input = img.insert_axis(Axis(0));
        let inputs = ort::inputs![ML_MODEL_INPUT_NAME => input.view()]?;

        let outputs = self.model.run(inputs)?;

        let output = outputs[ML_MODEL_OUTPUT_NAME].try_extract_tensor()?;
        let view = output.view();
        let output: ArrayView<f32, Dim<[usize; 2]>> = view.slice(s![0, 0, .., ..]);

        let image = postprocess_image(&output)?;

        let (original_width, original_height) = (original_img.width(), original_img.height());
        let resized = resize_rgba(&image, original_width, original_height)?;
        let img_buffer =
            ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(original_width, original_height, resized)
                .ok_or(anyhow!("Somehow image was resized to incorrect size"))?;
        let mask = DynamicImage::ImageRgba8(img_buffer);

        Ok(apply_mask(original_img, &mask))
    }
}

fn preprocess_image(image: &DynamicImage) -> anyhow::Result<Array3<f32>> {
    let img_vec = resize_rgba(image, ML_MODEL_IMAGE_WIDTH, ML_MODEL_IMAGE_HEIGHT)?;

    // Separate R, G, and B components
    let mut r_vec = Vec::with_capacity((ML_MODEL_IMAGE_WIDTH * ML_MODEL_IMAGE_HEIGHT) as usize);
    let mut g_vec = Vec::with_capacity((ML_MODEL_IMAGE_WIDTH * ML_MODEL_IMAGE_HEIGHT) as usize);
    let mut b_vec = Vec::with_capacity((ML_MODEL_IMAGE_WIDTH * ML_MODEL_IMAGE_HEIGHT) as usize);

    for chunk in img_vec.chunks(4) {
        r_vec.push(chunk[0]);
        g_vec.push(chunk[1]);
        b_vec.push(chunk[2]);
        // SKIP Alpha channel
    }

    // Concatenate R, G, and B vectors to form the correctly ordered vector
    let reordered_vec = [r_vec, g_vec, b_vec].concat();

    // Convert the resized image to a ndarray.
    let img_ndarray = Array3::from_shape_vec(
        (
            3,
            ML_MODEL_IMAGE_WIDTH as usize,
            ML_MODEL_IMAGE_HEIGHT as usize,
        ),
        reordered_vec,
    )?;

    // Convert to floating point and scale pixel values to [0, 1].
    let img_float: Array3<f32> = img_ndarray.mapv(|x| x as f32 / 255.0);

    // Normalize the image.
    Ok(normalize_image(&img_float))
}

fn normalize_image(img: &Array3<f32>) -> Array3<f32> {
    // The mean and std are applied across the channel dimension.
    let mean = Array3::from_elem((1, img.shape()[1], img.shape()[2]), 0.5);
    let std = Array3::from_elem((1, img.shape()[1], img.shape()[2]), 1.0);

    // Broadcasting the mean and std to match img dimensions and applying normalization.
    (img - &mean) / &std
}

fn postprocess_image(
    model_result: &ArrayView<f32, Dim<[usize; 2]>>,
) -> anyhow::Result<DynamicImage> {
    let ma = model_result
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or(anyhow!("Should be OK"))?;
    let mi = model_result
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or(anyhow!("Should be OK"))?;
    let result = (model_result.mapv(|x| x - mi) / (ma - mi)) * 255.0;

    let result_u8 = result.mapv(|x| x as u8).into_raw_vec();

    let mut imgbuf: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::new(ML_MODEL_IMAGE_WIDTH, ML_MODEL_IMAGE_HEIGHT);

    for (x, y, pixel) in imgbuf.enumerate_pixels_mut() {
        let index = (y * ML_MODEL_IMAGE_WIDTH + x) as usize;
        let value = result_u8[index];
        *pixel = Rgb([value, value, value]);
    }

    Ok(DynamicImage::ImageRgb8(imgbuf))
}

fn resize_rgba(
    img: &DynamicImage,
    target_width: u32,
    target_height: u32,
) -> anyhow::Result<Vec<u8>> {
    let width = NonZeroU32::new(img.width()).ok_or(anyhow!("Incorrect image width"))?;
    let height = NonZeroU32::new(img.height()).ok_or(anyhow!("Incorrect image height"))?;
    let mut src_image = fr::Image::from_vec_u8(
        width,
        height,
        img.to_rgba8().into_raw(),
        fr::PixelType::U8x4,
    )?;

    // Multiple RGB channels of source image by alpha channel
    let alpha_mul_div = fr::MulDiv::default();
    alpha_mul_div.multiply_alpha_inplace(&mut src_image.view_mut())?;

    // Create container for data of destination image
    let dst_width = NonZeroU32::new(target_width).ok_or(anyhow!("Incorrect target image width"))?;
    let dst_height =
        NonZeroU32::new(target_height).ok_or(anyhow!("Incorrect target image height"))?;
    let mut dst_image = fr::Image::new(dst_width, dst_height, src_image.pixel_type());

    // Get mutable view of destination image data
    let mut dst_view = dst_image.view_mut();

    // Create Resizer instance and resize source image
    // into buffer of destination image
    let mut resizer = fr::Resizer::new(fr::ResizeAlg::Convolution(fr::FilterType::Bilinear));
    resizer.resize(&src_image.view(), &mut dst_view)?;

    // Divide RGB channels of destination image by alpha
    alpha_mul_div.divide_alpha_inplace(&mut dst_view)?;

    Ok(dst_image.into_vec())
}

fn apply_mask(original_image: &DynamicImage, mask_image: &DynamicImage) -> DynamicImage {
    // Create a new transparent image
    let mut no_bg_image: RgbaImage = ImageBuffer::new(mask_image.width(), mask_image.height());

    // Manually apply the mask and overlay the original image
    for (x, y, pixel) in no_bg_image.enumerate_pixels_mut() {
        let orig_pixel = original_image.get_pixel(x, y);
        *pixel = Rgba([
            orig_pixel[0],
            orig_pixel[1],
            orig_pixel[2],
            mask_image.get_pixel(x, y)[0],
        ]);
    }
    DynamicImage::ImageRgba8(no_bg_image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageFormat;

    #[test]
    fn test_background_removal() {
        let original_img = image::open("assets/test.png").unwrap();

        let rmbg = Rmbg::new("assets/model.onnx").unwrap();

        let image_without_background = rmbg.remove_background(&original_img).unwrap();

        image_without_background
            .save_with_format("assets/no_bg.webp", ImageFormat::WebP)
            .unwrap();
    }
}
