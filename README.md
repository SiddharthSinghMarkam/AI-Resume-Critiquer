# AI Resume Critiquer 🤖

An intelligent web application that uses Google's Gemini AI to provide personalized and detailed feedback on resumes. Upload your PDF resume and receive comprehensive critiques to polish your professional documents.

## 🚀 Features

### 📄 PDF Resume Upload
Easily upload your resume in PDF format for analysis.

### 🎯 Customizable Feedback
Choose from multiple critique options tailored to your needs:

- **📊 Overall Critique** - Comprehensive review of your entire resume
- **🎯 Job-Specific Tailoring** - Optimize your resume for a specific job description
- **✍️ Grammar and Spelling Check** - Catch errors and improve language
- **💪 Strengths and Weaknesses Analysis** - Identify what works and what needs improvement
- **⚡ Action Verbs Analysis** - Enhance your bullet points with powerful verbs

### 🤖 AI-Powered Analysis
Utilizes Google's advanced `gemini-1.5-pro` model for intelligent, context-aware feedback.

### 🎨 Intuitive Interface
Clean, user-friendly interface built with Streamlit for seamless user experience.

## 📋 Prerequisites

Before you begin, ensure you have:

- **Google API Key** for Gemini API ([Get one from Google AI Studio](https://makersuite.google.com/app/apikey))
- **Python 3.8 or newer**

## 🛠️ Installation

### 1. Clone the Repository
git clone https://github.com/SiddharthSinghMarkam/your-project.git

### 2. Navigate to the project directory
   ```bash
   cd your-project
   ```

### 3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

### 4. Set up environment variables
   - Create a `.env` file in the project root
   - Add your Google API key:
     ```
     GOOGLE_API_KEY='your_api_key_here'
     ```

### Usage

1. Run the application
   
   streamlit run app.py
   
2. Open your browser - The app will automatically launch

3. Start critiquing - Upload your resume, select a critique type, and click "Get AI Feedback"

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Open an issue for bug reports or feature requests
- Submit a pull request with improvements
