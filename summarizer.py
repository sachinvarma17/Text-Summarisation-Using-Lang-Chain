from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline

# Load the transformers pipelines for summarization and question-answering
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")  # Specify a larger model
question_answering_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to split the text into chunks
def split_text(text, max_chunk_size=1024):
    words = text.split()
    return [" ".join(words[i:i + max_chunk_size]) for i in range(0, len(words), max_chunk_size)]


summarization_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Summarize the following text:
    {text}
    """
)

# Use HuggingFacePipeline in LangChain to integrate with transformers pipeline
summarization_llm = HuggingFacePipeline(pipeline=summarization_pipeline)

# Create the summarization chain
summarization_chain = LLMChain(llm=summarization_llm, prompt=summarization_prompt)

# Define the text to be summarized
input_text = """
The story of Baahubali spans across two films: Baahubali: The Beginning (2015) and Baahubali: The Conclusion (2017). Directed by S.S. Rajamouli, the epic tale is a mixture of fantasy, action, and drama, set in the fictional kingdom of Mahishmati. Here's a summary of the full story, including both parts:

### Baahubali: The Beginning (Part 1)

The film begins with a woman, Sivagami, struggling to save a baby from drowning in a river. She dies after saving the child, but a group of villagers from a tribal community finds and raises him. The child is named Shivudu (later known as Mahendra Baahubali). Growing up, Shivudu exhibits extraordinary strength and curiosity about what lies beyond the massive waterfall near his village.

One day, he discovers a mask that floats down from the waterfall and is enchanted by the idea of who it might belong to. Determined to find out, Shivudu climbs the waterfall and reaches the kingdom of Mahishmati. There, he meets a warrior named Avanthika, who is part of a rebel group trying to rescue their queen, **Devasena, held captive by the cruel king **Bhallaladeva. Shivudu falls in love with Avanthika and promises to rescue Devasena himself.

Shivudu manages to infiltrate Mahishmati's capital and frees Devasena, revealing his immense strength and skill. During this mission, it is discovered that Shivudu is actually Mahendra Baahubali, the son of the legendary king **Amarendra Baahubali. He is the rightful heir to the throne of Mahishmati.

The story then flashes back to when Amarendra Baahubali and his cousin Bhallaladeva were raised together by Sivagami, who served as the queen regent. Both cousins were trained as warriors and competed for the throne of Mahishmati. Amarendra, being noble and compassionate, won the hearts of the people and was declared king. However, Bhallaladeva, envious of his cousin, conspired against him.

### Baahubali: The Conclusion (Part 2)

In the second part, the story delves deeper into the backstory of Amarendra Baahubali. After being declared king, Amarendra is sent on a tour of the kingdom to understand his people better. During his journey, he meets Devasena, the princess of a neighboring kingdom, and falls in love with her. Amarendra proposes to Devasena, but this angers Bhallaladeva and his father, **Bijjaladeva, who have already planned to make her Bhallaladeva's wife.

Sivagami, unaware of Amarendra’s love for Devasena, promises her hand to Bhallaladeva. When Devasena rejects the proposal, Sivagami is insulted, leading to tension between her and Amarendra. Despite this, Amarendra marries Devasena and brings her to Mahishmati.

Bhallaladeva and Bijjaladeva continue to plot against Amarendra, manipulating Sivagami into believing that Amarendra is a threat to her authority. Sivagami, blinded by her loyalty to Bhallaladeva, orders Amarendra’s execution, believing it to be for the good of the kingdom. Kattappa, Amarendra's loyal general, is forced to carry out the order, killing Baahubali in one of the most tragic and shocking moments in the story. This is the famous revelation from the first film: “Kattappa ne Baahubali ko kyun maara?” ("Why did Kattappa kill Baahubali?").

Sivagami soon realizes her mistake when Bhallaladeva imprisons Devasena and takes over the throne, revealing his true nature. To protect Amarendra's infant son, Mahendra Baahubali, Sivagami escapes with the baby, leading to the events at the beginning of the first film.

In the present day, Mahendra Baahubali (Shivudu) learns the truth about his father’s death and Bhallaladeva’s tyranny. He rallies the people of Mahishmati and, with the help of Kattappa, Avanthika, and Devasena, declares war on Bhallaladeva. The final battle is an epic showdown between Mahendra and Bhallaladeva, with Mahendra ultimately emerging victorious.

Mahendra Baahubali is crowned the new king of Mahishmati, fulfilling his father’s legacy. Bhallaladeva is killed, and peace is restored to the kingdom. The film ends with a sense of closure, with justice served and the rightful ruler on the throne.

### Themes and Symbolism

Baahubali explores themes of loyalty, betrayal, family, and the responsibilities of leadership. The character of Amarendra Baahubali symbolizes the ideal king—just, brave, and compassionate—while Bhallaladeva represents greed and tyranny. The strong female characters, Sivagami and Devasena, play pivotal roles in the narrative, adding depth and strength to the story.

The story, with its grand scale, spectacular visuals, and epic storytelling, became one of the biggest blockbusters in Indian cinema, earning worldwide acclaim for its unique blend of fantasy and action."""

# Step 2: Check the input text length and split into chunks if necessary
text_chunks = split_text(input_text, max_chunk_size=512)  # Splitting into smaller chunks

# Summarize each chunk
summary_chunks = [summarization_chain.run(chunk) for chunk in text_chunks]
summary = " ".join(summary_chunks)
print("Summary:", summary)

# Step 3: Function to ask questions and get answers using question-answering model
def answer_question(context, question):
    qa_input = {
        "question": question,
        "context": context
    }
    return question_answering_pipeline(qa_input)['answer']

# Allow the user to ask questions about the text
while True:
    user_question = input("\nYou can now ask a question about the text (or type 'exit' to stop): ")

    if user_question.lower() == 'exit':
        break

    # Step 4: Get the answer to the user's question
    answer = answer_question(input_text, user_question)
    print(f"Answer: {answer}")