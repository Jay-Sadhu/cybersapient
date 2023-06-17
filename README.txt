My name is Sadhu Jay Vardhan, and I am currently in my third year of B.Tech, specializing in Artificial Intelligence at Mahindra University. My sincere gratitude for providing internship opportunities to students from Mahindra University. Cybersapient, being a renowned company specializing in cutting-edge technologies, is a dream workspace for any aspiring professional.

The task I choosen by me is to integrate the Movie Review System model with streamlit, I decided to leverage my expertise in Flask, as I felt more comfortable and proficient with it. I apologize for deviating from the specified requirement of using Streamlit, but given the limited time and my familiarity with Flask, I believed it would allow me to deliver the project within the given deadline.

During the deployment process, I encountered some challenges while using Zeet and PythonAnywhere. Unfortunately, my attempts to resolve these issues were not successful. In order to provide you with a clear understanding of the output, I have recorded a video documenting the website built using flask for better understanding on how it turned out to be.

-----------------Please watch the video---------------------


------------------------------------------------------------------------------------------------------------------------------------------

Regarding the Project:

Sentiment Analysis with Summarization and Translation :
This repository contains code for performing sentiment analysis on movie reviews using logistic regression, along with additional features for text summarization and translation.

Dataset:
The code uses the IMDB movie review dataset, which is split into training and testing sets. The dataset is loaded using pandas and preprocessed to remove stop words and non-alphabetic characters.

Sentiment Analysis:
The sentiment analysis model is trained using logistic regression. The text data is transformed into numerical features using TF-IDF vectorization. The trained model is then used to predict the sentiment (positive or negative) for the test dataset. The accuracy of the model is calculated using the test dataset.

Summarization:
The code includes a text summarization function that uses the LSA (Latent Semantic Analysis) algorithm from the sumy library. The function takes a text as input and summarizes it into a single sentence.

Translation:
The code also provides a translation function using the Google Translate API. The function takes a text and a target language as input and translates the text into the specified language.

Steps to run:
1. Install the required libraries: pandas, nltk, scikit-learn, sumy, and googletrans.
2. Download the IMDB movie review dataset and update the file path in the code accordingly.
3. Run the code to preprocess the data, train the sentiment analysis model, and calculate the accuracy.
4. Modify the `summarize_text` and `translate_text` functions as needed for your use case.
5. Use the `summarize_text` function to summarize a text and the `translate_text` function to translate a text into a different language.
6. Update the `example_text` variable with your own text for sentiment analysis, summarization, and translation.


The code requires the following libraries:
- pandas: for data manipulation and analysis
- nltk: for natural language processing tasks
- scikit-learn: for machine learning algorithms
- sumy: for text summarization
- googletrans: for translation using the Google Translate API


------------------------------------------------------------------------------------------------------------------------------------------

Inorder to run the flask application:

This Flask application performs sentiment analysis on movie reviews using a logistic regression model. It provides a user interface where users can input a movie review and get the sentiment (positive or negative) predicted by the model.


flask code(ofcourse the dataset has also been used as it is the part of the architecture), html and css code has been used to create the website.


Steps to create the virtual environment and run the flask command:
virtualenv env
.\env\Scripts\activate
python .\app.py

File Structure:
- app.py: The main Flask application file.
- templates/cs_index.html: The HTML template for the user interface.
- static/cs_st.css: The CSS file for styling the user interface.

Usage:
1. Install the required libraries: pandas, nltk, scikit-learn, and Flask.
2. Download the IMDB movie review dataset and update the file path in the code (data = pd.read_csv(r'IMDB Dataset.csv')) to match the location of your dataset.
3. Run the Flask application using python app.py command.
4. Access the application in your browser at http://localhost:5000.
5. Enter a movie review in the provided text area and click "Submit".
6. The application will predict the sentiment of the review and display the result.


