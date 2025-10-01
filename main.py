import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class StrokeDatasetEDA:
    def __init__(self, data_dir="train"):
        """
        Initialize the EDA class with the dataset directory
        """
        self.data_dir = Path(data_dir)
        self.csv_file = self.data_dir / "_classes.csv"
        self.df = None
        self.class_names = [" Hemorrhagic", " Ischaemic", " NORMAL", " Unlabeled"]
        self.class_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    def load_data(self):
        """Load the dataset from CSV file"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"✅ Successfully loaded dataset with {len(self.df)} samples")
            return True
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return False

    def basic_info(self):
        """Display basic information about the dataset"""
        print("=" * 60)
        print("📊 STROKE DATASET - BASIC INFORMATION")
        print("=" * 60)

        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Images: {len(self.df)}")
        print(f"Features: {list(self.df.columns)}")
        print()

        print("📋 Column Information:")
        print(self.df.info())
        print()

        print("📈 First 5 rows:")
        print(self.df.head())
        print()

        print("🔍 Missing Values:")
        missing_data = self.df.isnull().sum()
        print(missing_data)
        print()

    def class_distribution(self):
        """Analyze class distribution"""
        print("=" * 60)
        print("📊 CLASS DISTRIBUTION ANALYSIS")
        print("=" * 60)

        # Calculate class counts
        class_counts = {}
        for class_name in self.class_names:
            class_counts[class_name] = self.df[class_name].sum()

        print("Class Distribution:")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{class_name}: {count} samples ({percentage:.2f}%)")

        print()

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Bar plot
        bars = ax1.bar(
            class_counts.keys(), class_counts.values(), color=self.class_colors
        )
        ax1.set_title("Class Distribution - Bar Chart", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Classes", fontsize=12)
        ax1.set_ylabel("Number of Samples", fontsize=12)
        ax1.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values()):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 10,
                f"{count}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Pie chart
        wedges, texts, autotexts = ax2.pie(
            class_counts.values(),
            labels=class_counts.keys(),
            colors=self.class_colors,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax2.set_title("Class Distribution - Pie Chart", fontsize=14, fontweight="bold")

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_fontweight("bold")
            autotext.set_fontsize(10)

        plt.tight_layout()
        plt.show()

        return class_counts

    def filename_analysis(self):
        """Analyze filename patterns"""
        print("=" * 60)
        print("📁 FILENAME ANALYSIS")
        print("=" * 60)

        # Extract patterns from filenames
        self.df["file_extension"] = self.df["filename"].str.extract(r"\.([^.]+)$")
        self.df["file_prefix"] = self.df["filename"].str.extract(r"^([^_]+)")

        print("File Extensions:")
        print(self.df["file_extension"].value_counts())
        print()

        print("File Prefixes (Top 10):")
        print(self.df["file_prefix"].value_counts().head(10))
        print()

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # File extensions
        ext_counts = self.df["file_extension"].value_counts()
        ax1.pie(
            ext_counts.values, labels=ext_counts.index, autopct="%1.1f%%", startangle=90
        )
        ax1.set_title("File Extensions Distribution", fontsize=12, fontweight="bold")

        # Top file prefixes
        prefix_counts = self.df["file_prefix"].value_counts().head(10)
        bars = ax2.bar(range(len(prefix_counts)), prefix_counts.values, color="skyblue")
        ax2.set_title("Top 10 File Prefixes", fontsize=12, fontweight="bold")
        ax2.set_xlabel("File Prefixes")
        ax2.set_ylabel("Count")
        ax2.set_xticks(range(len(prefix_counts)))
        ax2.set_xticklabels(prefix_counts.index, rotation=45, ha="right")

        # Add value labels
        for bar, count in zip(bars, prefix_counts.values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 5,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Filename length distribution
        filename_lengths = self.df["filename"].str.len()
        ax3.hist(
            filename_lengths, bins=30, color="lightgreen", alpha=0.7, edgecolor="black"
        )
        ax3.set_title("Filename Length Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("Filename Length (characters)")
        ax3.set_ylabel("Frequency")
        ax3.axvline(
            filename_lengths.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {filename_lengths.mean():.1f}",
        )
        ax3.legend()

        # Class distribution by file prefix (top 5 prefixes)
        top_prefixes = self.df["file_prefix"].value_counts().head(5).index
        prefix_class_data = []
        for prefix in top_prefixes:
            prefix_df = self.df[self.df["file_prefix"] == prefix]
            class_counts = [
                prefix_df[class_name].sum() for class_name in self.class_names
            ]
            prefix_class_data.append(class_counts)

        x = np.arange(len(top_prefixes))
        width = 0.2

        for i, class_name in enumerate(self.class_names):
            values = [data[i] for data in prefix_class_data]
            ax4.bar(
                x + i * width,
                values,
                width,
                label=class_name,
                color=self.class_colors[i],
            )

        ax4.set_title(
            "Class Distribution by Top 5 File Prefixes", fontsize=12, fontweight="bold"
        )
        ax4.set_xlabel("File Prefixes")
        ax4.set_ylabel("Count")
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels(top_prefixes, rotation=45, ha="right")
        ax4.legend()

        plt.tight_layout()
        plt.show()

    def image_analysis(self, sample_size=100):
        """Analyze image properties (size, format, etc.)"""
        print("=" * 60)
        print("🖼️ IMAGE ANALYSIS")
        print("=" * 60)

        # Sample images for analysis (to avoid processing all images)
        sample_indices = np.random.choice(
            len(self.df), min(sample_size, len(self.df)), replace=False
        )
        sample_df = self.df.iloc[sample_indices]

        image_properties = []

        print(f"Analyzing {len(sample_df)} sample images...")

        for idx, row in sample_df.iterrows():
            img_path = self.data_dir / row["filename"]
            try:
                # Load image with OpenCV
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width, channels = img.shape
                    file_size = img_path.stat().st_size / 1024  # KB

                    image_properties.append(
                        {
                            "filename": row["filename"],
                            "width": width,
                            "height": height,
                            "channels": channels,
                            "file_size_kb": file_size,
                            "aspect_ratio": width / height,
                        }
                    )
            except Exception as e:
                print(f"Error processing {row['filename']}: {e}")

        if not image_properties:
            print("❌ No images could be processed")
            return

        # Convert to DataFrame
        img_df = pd.DataFrame(image_properties)

        print("📊 Image Properties Summary:")
        print(f"Images analyzed: {len(img_df)}")
        print(f"Average width: {img_df['width'].mean():.1f} pixels")
        print(f"Average height: {img_df['height'].mean():.1f} pixels")
        print(f"Average file size: {img_df['file_size_kb'].mean():.1f} KB")
        print(f"Average aspect ratio: {img_df['aspect_ratio'].mean():.2f}")
        print()

        # Create visualizations
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Width distribution
        ax1.hist(
            img_df["width"], bins=20, color="lightblue", alpha=0.7, edgecolor="black"
        )
        ax1.set_title("Image Width Distribution", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Width (pixels)")
        ax1.set_ylabel("Frequency")
        ax1.axvline(
            img_df["width"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {img_df["width"].mean():.1f}',
        )
        ax1.legend()

        # Height distribution
        ax2.hist(
            img_df["height"], bins=20, color="lightgreen", alpha=0.7, edgecolor="black"
        )
        ax2.set_title("Image Height Distribution", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Height (pixels)")
        ax2.set_ylabel("Frequency")
        ax2.axvline(
            img_df["height"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {img_df["height"].mean():.1f}',
        )
        ax2.legend()

        # File size distribution
        ax3.hist(
            img_df["file_size_kb"],
            bins=20,
            color="lightcoral",
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_title("File Size Distribution", fontsize=12, fontweight="bold")
        ax3.set_xlabel("File Size (KB)")
        ax3.set_ylabel("Frequency")
        ax3.axvline(
            img_df["file_size_kb"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {img_df["file_size_kb"].mean():.1f} KB',
        )
        ax3.legend()

        # Aspect ratio distribution
        ax4.hist(
            img_df["aspect_ratio"],
            bins=20,
            color="lightyellow",
            alpha=0.7,
            edgecolor="black",
        )
        ax4.set_title("Aspect Ratio Distribution", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Aspect Ratio (Width/Height)")
        ax4.set_ylabel("Frequency")
        ax4.axvline(
            img_df["aspect_ratio"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {img_df["aspect_ratio"].mean():.2f}',
        )
        ax4.legend()

        plt.tight_layout()
        plt.show()

        return img_df

    def class_correlation_analysis(self):
        """Analyze correlations between classes"""
        print("=" * 60)
        print("🔗 CLASS CORRELATION ANALYSIS")
        print("=" * 60)

        # Calculate correlation matrix for class columns
        class_cols = [col for col in self.class_names if col in self.df.columns]
        correlation_matrix = self.df[class_cols].corr()

        print("Correlation Matrix:")
        print(correlation_matrix.round(3))
        print()

        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Class Correlation Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.show()

        # Check for samples with multiple classes
        multi_class_samples = self.df[class_cols].sum(axis=1)
        print(f"Samples with single class: {(multi_class_samples == 1).sum()}")
        print(f"Samples with multiple classes: {(multi_class_samples > 1).sum()}")
        print(f"Samples with no class: {(multi_class_samples == 0).sum()}")
        print()

    def sample_images_display(self, samples_per_class=4):
        """Display sample images from each class"""
        print("=" * 60)
        print("🖼️ SAMPLE IMAGES DISPLAY")
        print("=" * 60)

        fig, axes = plt.subplots(
            len(self.class_names),
            samples_per_class,
            figsize=(samples_per_class * 3, len(self.class_names) * 3),
        )

        if len(self.class_names) == 1:
            axes = axes.reshape(1, -1)

        for class_idx, class_name in enumerate(self.class_names):
            if class_name not in self.df.columns:
                continue

            # Get samples for this class
            class_samples = self.df[self.df[class_name] == 1]

            if len(class_samples) == 0:
                continue

            # Sample random images
            sample_indices = np.random.choice(
                len(class_samples),
                min(samples_per_class, len(class_samples)),
                replace=False,
            )

            for sample_idx, img_idx in enumerate(sample_indices):
                img_path = self.data_dir / class_samples.iloc[img_idx]["filename"]

                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        axes[class_idx, sample_idx].imshow(img_rgb)
                        axes[class_idx, sample_idx].set_title(
                            f'{class_name}\n{class_samples.iloc[img_idx]["filename"]}',
                            fontsize=8,
                        )
                        axes[class_idx, sample_idx].axis("off")
                    else:
                        axes[class_idx, sample_idx].text(
                            0.5,
                            0.5,
                            "Image\nNot Found",
                            ha="center",
                            va="center",
                            transform=axes[class_idx, sample_idx].transAxes,
                        )
                        axes[class_idx, sample_idx].axis("off")
                except Exception as e:
                    axes[class_idx, sample_idx].text(
                        0.5,
                        0.5,
                        f"Error\n{str(e)[:20]}",
                        ha="center",
                        va="center",
                        transform=axes[class_idx, sample_idx].transAxes,
                    )
                    axes[class_idx, sample_idx].axis("off")

        plt.suptitle("Sample Images from Each Class", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Generate a comprehensive EDA report"""
        print("🚀 Starting Comprehensive EDA Analysis...")
        print()

        # Load data
        if not self.load_data():
            return

        # Basic information
        self.basic_info()

        # Class distribution
        class_counts = self.class_distribution()

        # Filename analysis
        self.filename_analysis()

        # Image analysis
        self.image_analysis()

        # Class correlation analysis
        self.class_correlation_analysis()

        # Sample images display
        self.sample_images_display()

        print("=" * 60)
        print("✅ EDA ANALYSIS COMPLETED!")
        print("=" * 60)

        # Summary
        print("📋 SUMMARY:")
        print(f"• Total samples: {len(self.df)}")
        print(f"• Classes: {', '.join(self.class_names)}")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"• {class_name}: {count} samples ({percentage:.1f}%)")
        print("• Dataset appears to be ready for machine learning tasks")


def main():
    """Main function to run the EDA"""
    print("🧠 STROKE DATASET EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Initialize EDA
    eda = StrokeDatasetEDA("train")

    # Run comprehensive analysis
    eda.generate_report()


if __name__ == "__main__":
    main()
