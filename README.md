Here is a well-structured and informative README file for your project:

---

# Object Detection and Defect Identification with AI 🤖

This project leverages AI to detect mechanical components in images and identify any defects using a deep learning model and Generative AI. It also provides functionality for storing and querying images, PDFs, and common PDFs related to each mechanical component.

## 📝 Project Overview

The **Object Detection and Defect Identification** project aims to:

1. Upload images, relevant PDFs, and common PDFs into an SQLite database.
2. Use AI to analyze uploaded images, detect components, and identify any defects.
3. Provide download links for relevant documents associated with detected components.
4. Generate search links to Amazon for defected components.

## 🚀 Features

- **Image Upload**: Users can upload images and related PDFs for component analysis.
- **AI-powered Analysis**: Uses Google Generative AI to detect mechanical components in images.
- **Database Integration**: Stores images and PDFs in an SQLite database for easy retrieval.
- **Defect Detection**: Identifies defects in mechanical components by comparing detected components against the expected list.
- **Amazon Integration**: Provides search links for defective components on Amazon.
- **PDF Handling**: Loads and processes PDFs to extract relevant data for querying.
- **Interactive Interface**: Built with Streamlit, offering a simple and intuitive interface for users.

## 🛠️ Installation

To get started with this project, clone the repository and install the required dependencies.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/your-repository-name.git
   cd your-repository-name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 💡 How It Works

### 1. **Database Creation**
The `Database_Creation_Code.py` script enables the upload of images and PDFs to an SQLite database. The data is stored as binary blobs for efficient handling and retrieval.

### 2. **Object Detection & Defect Identification**
The `Main.py` script is the core of the object detection and defect identification system. It uses the following steps:

- **Image Upload**: The user uploads an image for analysis.
- **Image Comparison**: The uploaded image is compared with stored images in the database using Structural Similarity Index (SSIM).
- **Component Detection**: The Google Generative AI (Gemini) model is used to analyze the image and detect mechanical components.
- **Defect Detection**: If any components are missing from the detected list, they are flagged as defects.
- **Amazon Links**: For each defect, a clickable Amazon search link is generated to help users find replacement parts.

### 3. **Text Retrieval from PDFs**
The system also processes PDFs associated with the images and provides information on mechanical components. It uses `Langchain` to extract and retrieve relevant information from the PDFs.

## 🧑‍💻 Usage

1. **Start the Streamlit app**:
   ```bash
   streamlit run Main.py
   ```

2. **Upload an Image**: The app will prompt you to upload an image to analyze.
3. **Download Relevant PDFs**: Once a similar image is found, you can download the relevant and common PDFs associated with the detected components.
4. **Defect Detection**: The system will analyze the image, detect components, and identify defects, showing a list of detected components and providing Amazon links for any defects.

## 🔧 Requirements

- **Python 3.x**
- **Required Libraries**: All necessary libraries are listed in the `requirements.txt` file:
  - `streamlit`
  - `numpy`
  - `opencv-python`
  - `scikit-image`
  - `langchain_community`
  - `langchain_huggingface`
  - `langchain_google_genai`
  - `transformers`
  - `pypdf`
  - `chromadb`

## ⚙️ Technologies Used

- **Streamlit**: For building the interactive web application.
- **Google Generative AI (Gemini)**: For analyzing images and detecting mechanical components.
- **SQLite**: For storing images, PDFs, and related data.
- **Chroma**: For creating vector databases and handling text data.
- **OpenCV**: For image processing and similarity comparison.
- **Amazon API Integration**: For linking defective components to Amazon product searches.

## 📂 Project Structure

```
├── Database_Creation_Code.py       # Script for creating and populating the SQLite database
├── Main.py                         # Core object detection and defect identification script
├── requirements.txt                # List of dependencies
└── chroma_db/                      # Directory for storing Chroma vector databases
└── chroma_separatekb_db/           # Directory for storing separate Chroma vector databases
```

## 🌐 Future Enhancements

- **Integration with more AI models**: To enhance defect detection and image analysis.
- **Real-time Object Detection**: Stream real-time video feeds for defect identification.
- **User Management**: Add user authentication for data privacy and control.

