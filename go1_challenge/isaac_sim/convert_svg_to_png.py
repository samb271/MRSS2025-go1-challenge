import os
from pathlib import Path

try:
    from PIL import Image
    import cairosvg

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL and cairosvg not available. Please install with:")
    print("pip install pillow cairosvg")


def convert_svg_to_png(svg_dir: str, output_dir: str, size: int = 512):
    """Convert SVG files to PNG format for USD texture usage."""

    if not PIL_AVAILABLE:
        return False

    svg_path = Path(svg_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    svg_files = list(svg_path.glob("*.svg"))

    if not svg_files:
        print(f"No SVG files found in {svg_path}")
        return False

    print(f"Converting {len(svg_files)} SVG files to PNG...")

    for svg_file in svg_files:
        # Generate output filename
        png_filename = f"aruco_tag_{svg_file.stem}.png"
        png_file = output_path / png_filename

        try:
            # Convert SVG to PNG using cairosvg
            cairosvg.svg2png(
                url=str(svg_file),
                write_to=str(png_file),
                output_width=size,
                output_height=size,
            )
            print(f"Converted {svg_file.name} -> {png_filename}")

        except Exception as e:
            print(f"Error converting {svg_file.name}: {e}")

    return True


if __name__ == "__main__":
    # Convert SVG files to PNG
    svg_directory = Path(__file__).parent / "assets" / "tags"
    png_directory = Path(__file__).parent / "assets" / "textures" / "aruco_tags"

    success = convert_svg_to_png(str(svg_directory), str(png_directory))

    if success:
        print("SVG to PNG conversion completed!")
    else:
        print("Conversion failed. Please install required packages.")
