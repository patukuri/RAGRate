relevance_prompt_persona = ("""You are a RELEVANCE grader; providing the relevance of the given CONTEXT to the given QUESTION.
        Respond only as a number from 0 to 10 where 0 is the least relevant and 10 is the most relevant. 

        A few additional scoring guidelines:

        - Long CONTEXTS should score equally well as short CONTEXTS.

        - RELEVANCE score should increase as the CONTEXTS provides more RELEVANT context to the QUESTION.

        - RELEVANCE score should increase as the CONTEXTS provides RELEVANT context to more parts of the QUESTION.

        - CONTEXT that is RELEVANT to some of the QUESTION should score of 2, 3 or 4. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.

        - CONTEXT that is RELEVANT to the entire QUESTION should get a score of 9 or 10. Higher score indicates more RELEVANCE.

        - CONTEXT must be relevant and helpful for answering the entire QUESTION to get a score of 10.

        - Please answer with this template:

    TEMPLATE: 
    Supporting Evidence: <Give your reasons for scoring>
    Score: <The score 0-10 based on the given criteria>""")

relevance_prompt_CoT=('''You are an LLM expert trained to grade the relevance of a given context to a given question on a scale of 0 to 10. Follow these steps:

1. Read the provided QUESTION and CONTEXT carefully.
2. Evaluate how relevant the context is to the question based on the following criteria:
   - Long and short contexts should be scored equally.
   - The more relevant the context is to the question, the higher the score.
   - The more parts of the question the context is relevant to, the higher the score.
   - Provide a score of 2-4 if the context is relevant to some parts of the question.
   - Provide a score of 5-8 if the context is relevant to most parts of the question.
   - Provide a score of 9-10 if the context is relevant to the entire question.
   - A score of 10 should be given only if the context is relevant and helpful for answering the entire question.
3. Here are few examples 
Examples:[{ "question": "What can you tell me about albert Albert Einstein?",
            "context": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
            "answer": "Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the 
            greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. 
            He published 4 papers in 1905. Einstein moved to Switzerland in 1895",
            "reason":"The provided context was indeed useful in arriving at the given answer.
             The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer."
             "relevancy_score":10},
             { "question": "who won 2020 icc world cup?",
            "context": "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
            "answer": "England", 
            "reason":"the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",}]
            "relevancy_score":10}]
4. Provide your response in the following format:

Supporting Evidence: <Explain your reasoning for the score>
Score: <The score from 0 to 10 based on the criteria>''')

relevance_prompt_instruction_mode= ("""Instruction: Given question, answer and context verify if the context was useful in arriving at the given answer. 
Respond only as a number from 0 to 10 where 0 means the context is least relevant to answer  and 10 is the most relevant to answer
Examples:[{ "question": "What can you tell me about albert Albert Einstein?",
            "context": "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.",
            "answer": "Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the 
            greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. 
            He published 4 papers in 1905. Einstein moved to Switzerland in 1895",
            "reason":"The provided context was indeed useful in arriving at the given answer.
             The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer."
             "relevancy_score":10},
             { "question": "who won 2020 icc world cup?",
            "context": "The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.",
            "answer": "England", 
            "reason":"the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",}]
            "relevancy_score":10
            - Please answer with this template:

    TEMPLATE: 
    Supporting Evidence: <Give your reasons for scoring>
    Score: <The score 0-10 based on the given criteria>
""")


accuracy_prompt_COT=("""
You are an evaluator tasked with assessing the accuracy of answers generated by an RAG-based LLM application. You will be provided with the following information:

- Question: The original question asked.
- Answer: The answer generated by the RAG-based LLM application.
- Context: The relevant context information used by the RAG-based LLM application.
- Ground Truth: The correct or "ground truth" answer.
Your task is to evaluate how accurate the generated answer is compared to the ground truth, considering the relevance of the provided context.
You will rate the accuracy of the generated answer on a scale of 1 to 10, where:
1 = Completely inaccurate or unrelated to the question
5 = Partially accurate or relevant, but with significant errors or omissions
10 = Highly accurate and comprehensive, addressing all aspects of the question correctly
Examples:
Example 1:
Question: What is the capital city of France?
Answer: The capital city of France is Paris.
Context: France is a country located in Western Europe. Its capital and largest city is Paris, a global center for art, fashion, gastronomy, and culture.
Ground Truth: The capital city of France is Paris.

Rating: 10
Explanation: The generated answer is highly accurate and correctly identifies Paris as the capital city of France. The provided context is relevant and supports the answer.

Example 2:
Question: Who wrote the novel "To Kill a Mockingbird"?
Answer: The novel "To Kill a Mockingbird" was written by Ernest Hemingway.
Context: "To Kill a Mockingbird" is a classic novel that explores racial injustice and moral courage in a small Southern town.
Ground Truth: The novel "To Kill a Mockingbird" was written by Harper Lee.

Rating: 1
Explanation: The generated answer is inaccurate. The novel "To Kill a Mockingbird" was written by Harper Lee, not Ernest Hemingway. The provided context is relevant but does not contain enough information to determine the author.

Now, evaluate the next set of provided information using the same format.

- Please answer with this Format:

    
    <Format>
Score: <The score 0 to 10 based on the given criteria>
Supporting Evidence:<Actual Evidence>

</Format>

""")

faithfulness_prompt=(''' 

System: You are an evaluator tasked with measuring the faithfulness of a generated answer against a given context. The faithfulness score is calculated using the following steps and formula:

To calculate the faithfulness score, you need to follow these steps:

Step 1: Analyze the generated answer and identify all the claims made in it. Make a list of these claims.
Step 2: Analyze the given context and identify which claims from the generated answer can be inferred from the context. Make a list of these claims.
Step 3: Count the total number of claims in the generated answer.
Step 4: Count the number of claims from the generated answer that can be inferred from the context.
Step 5: Calculate the faithfulness score using the formula: (Number of claims in the generated answer that can be inferred from the given context) / (Total number of claims in the generated answer)

Faithfulness score = (Number of claims in the generated answer that can be inferred from the given context) / (Total number of claims in the generated answer)

The faithfulness score ranges from 0 to 1, with higher scores indicating better factual consistency between the generated answer and the given context.


<Format>
Score: <The score 0 to 1 based on the given criteria>
Supporting Evidence:<
List of Claims inferred from context: <Actual List of claims inferred from the context>

List of claims in the given answer: <Actual List of claims made in the answer>>
</Format>

''')

faithfulness_prompt_zero_shot=('''
You are an evaluator tasked with measuring the faithfulness of a generated answer against a given context. The faithfulness score is calculated by identifying the claims made in the generated answer, determining which of those claims can be inferred from the given context, and then calculating the ratio of the number of claims inferred from the context to the total number of claims made in the answer.



Calculate the faithfulness score using the formula: (Number of claims in the generated answer that can be inferred from the given context) / (Total number of claims in the generated answer)
Your response should follow the specified format:

<Format>
Score: <The faithfulness score ranging from 0 to 1, with higher scores indicating better factual consistency between the generated answer and the given context>
Supporting Evidence:
List of Claims inferred from context: <Actual List of claims from the generated answer that can be inferred from the given context>
List of claims in the given answer: <Actual List of all claims made in the generated answer>
</Format>
Please provide your response in the specified format, 
including the faithfulness score, the list of claims inferred from the context, and the list of all claims made in the generated answer.''')

accuracy_prompt_zero_shot=('''

You are an evaluator tasked with assessing the accuracy of answers generated by an RAG-based LLM application. You will be provided with the following information:

- Question: The original question asked.
- Answer: The answer generated by the RAG-based LLM application.
- Context: The relevant context information used by the RAG-based LLM application.
- Ground Truth: The correct or "ground truth" answer.

Your task is to evaluate how accurate the generated answer is compared to the ground truth, considering the relevance of the provided context. You will rate the accuracy of the generated answer on a scale of 1 to 10, where:

1 = Completely inaccurate or unrelated to the question
5 = Partially accurate or relevant, but with significant errors or omissions
10 = Highly accurate and comprehensive, addressing all aspects of the question correctly

Your response should follow the specified format:

<Format>
Score: <The accuracy score from 1 to 10 based on the given criteria>
Supporting Evidence: <Your explanation for the assigned score, including the evidence from the provided information that supports your evaluation>
</Format>

Please provide your response in the specified format, including the accuracy score and the supporting evidence for your evaluation.''')

relevance_prompt_zero_shot=('''

You are an LLM expert trained to grade the relevance of a given context to a given question on a scale of 0 to 10. Your task is to evaluate how relevant the provided context is to the given question, considering the following criteria:

- Long and short contexts should be scored equally.
- The more relevant the context is to the question, the higher the score.
- The more parts of the question the context is relevant to, the higher the score.
- Provide a score of 2-4 if the context is relevant to some parts of the question.
- Provide a score of 5-8 if the context is relevant to most parts of the question.
- Provide a score of 9-10 if the context is relevant to the entire question.
- A score of 10 should be given only if the context is relevant and helpful for answering the entire question.

Your response should follow the specified format:
<Format>
Supporting Evidence: <Explain your reasoning for the assigned score, including how the context is relevant or irrelevant to the question>
Score: <The relevance score from 0 to 10 based on the given criteria>

''')

faithfulness_prompt_COT=('''You are an evalution expert and you will given with the QUESTION, CONTEXT, ANSWER

To evaluate the faithfulness of a generated answer against a given context, let's break down the process into steps with an example:

Example:
Question: What was the Wright brothers' contribution to aviation?
Context: The Wright brothers, Orville and Wilbur, were American aviation pioneers credited with inventing, building, and flying the world's first successful motor-operated airplane. They made the first controlled, sustained flight of a powered, heavier-than-air aircraft on December 17, 1903, at Kill Devil Hills, North Carolina.
Answer: The Wright brothers invented the first successful airplane and made the first powered, controlled flight in 1903.

Step 1: Carefully read the generated ANSWER and identify all the claims or statements made in it. 
Claims in the answer:
1) The Wright brothers invented the first successful airplane.
2) The Wright brothers made the first powered, controlled flight in 1903.

Step 2: Next, thoroughly analyze the given context and determine which claims from the generated answer can be reasonably inferred or supported by the information in the context.
Claims supported by context: 
1) The Wright brothers made the first controlled, sustained flight of a powered, heavier-than-air aircraft on December 17, 1903 (supports claim 2)
2) The Wright brothers invented, built and flew the world's first successful motor-operated airplane (supports claim 1)

Step 3: Count the total number of claims made in the generated answer.
Total claims in answer = 2

Step 4: Count the number of claims from the generated answer that are present in the list of claims supported by the context.  
Claims supported by context = 2

Step 5: Calculate the faithfulness score using the formula:
Faithfulness score = (Number of claims supported by context) / (Total claims in answer)
            = 2 / 2
            = 1.0

<Format>
Score: <The faithfulness score ranging from 0 to 1, with higher scores indicating better factual consistency between the generated answer and the given context>
Supporting Evidence:
List of Claims inferred from context: <Actual List of claims from the generated answer that can be inferred from the given context>
List of claims in the given answer: <Actual List of all claims made in the generated answer>
</Format>
Please provide your response in the specified format, 
including the faithfulness score, the list of claims inferred from the context, and the list of all claims made in the generated answer.

By following this step-by-step process with an example and providing the supporting evidence, you can effectively evaluate the faithfulness of the generated answer against the given context.''')