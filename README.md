
# **News Article Categorization System with BERT Fine-Tuning and SQLAlchemy**

## **Project Overview**

This project demonstrates a robust pipeline for collecting, storing, and classifying news articles into predefined categories. By combining Python's SQLAlchemy for data management and the Hugging Face Transformers library for fine-tuning a DistilBERT model, the system automates the entire workflow from data ingestion to model inference.

The main features of this project include:
- **Data Management**: SQLAlchemy is used to interact with a MySQL database, allowing efficient storage and retrieval of articles.
- **BERT Model Fine-Tuning**: The project utilizes a pre-trained DistilBERT model, fine-tuned on custom data to classify news articles into categories based on their content.
- **Complete ML Pipeline**: The pipeline covers data preprocessing, model training, evaluation, and inference, ensuring seamless integration between the database and the machine learning components.

## **Technologies and Tools**
- **Python**: The core language used for implementing both the SQLAlchemy and machine learning components.
- **SQLAlchemy**: An ORM (Object Relational Mapper) used for database management and communication with MySQL.
- **MySQL**: A relational database system used to store news articles and their associated metadata.
- **Hugging Face Transformers**: For fine-tuning the DistilBERT model on custom text data for news categorization.
- **Pandas**: For data manipulation and preprocessing.
- **Scikit-learn**: For splitting datasets and evaluating model performance.

## **Features**
1. **Data Ingestion**: News articles are fetched from various RSS feeds and processed before being stored in a MySQL database.
2. **Data Preprocessing**: The project cleans, de-duplicates, and tokenizes the article text (headlines and descriptions) for training.
3. **Model Fine-Tuning**: The DistilBERT model is fine-tuned using a sequence classification head, optimized with training strategies like gradient accumulation and weight decay.
4. **Evaluation and Inference**: The model is evaluated on a validation dataset, and it predicts the category of unseen news articles with high accuracy.
5. **Scalable Design**: The project is designed to be modular, allowing easy extension to handle more data sources or to replace the model with another transformer-based architecture.

## **Installation Instructions**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/news-article-categorization.git
   cd news-article-categorization
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Set Up MySQL Database**:
   Ensure you have MySQL installed and running on your system. Create a new database using the following SQL command:
   
4. **Run the Data Collection and Storage**:
   To fetch articles and store them in the database, run:
   

5. **Fine-Tune the BERT Model**:
   Once the data is ingested and cleaned, run the following script to fine-tune the BERT model on your dataset:
  




## **Dataset**
The dataset consists of news articles fetched from various RSS feeds. It contains fields such as:
- **headline**: The headline of the news article.
- **short_description**: A brief description of the article.
- **category**: The category or topic to which the article belongs (e.g., WORLD NEWS, POLITICS, SPORTS).

## **Model Overview**
The fine-tuning process leverages Hugging Face's DistilBERT model, a lighter version of BERT, optimized for faster performance with minimal loss in accuracy. The model is trained on the news article dataset using a sequence classification head to predict the article category based on its headline and description.

### Training Parameters:
- **Learning Rate**: 2e-5
- **Batch Size**: 8
- **Epochs**: 10
- **Evaluation Strategy**: End of every epoch
- **Weight Decay**: 0.01

## **Results**
 **Model Inference**: The model successfully predicted categories for unseen articles, providing meaningful insights into its performance.

## **Future Enhancements**
- **Adding More Categories**: Expand the dataset to include more nuanced categories.
- **Integrating New Models**: Experiment with other transformer models like RoBERTa or GPT for further improvements.
- **Real-Time Inference**: Deploy the model as an API to provide real-time news categorization.


