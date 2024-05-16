import os
from asyncio import exceptions
import os
import openai
import pandas as pd
import re

from anthropic import Anthropic
from groq import Groq
import anthropic
import matplotlib.pyplot as plt
import time

from openai import OpenAI

from src.ragrate.metrics import prompts

api_base = "https://api.openai.com/v1"
openai.api_key =
os.environ["OPENAI_API_KEY"] = openai.api_key
os.environ["OPENAI_API_BASE"] = api_base


class LLMEvaluator:
    def __init__(self, anthropic_key, api_key, open_ai_api_key, model_names, max_retries=10, retry_delay=40):
        self.anthropic_key = anthropic_key
        self.api_key = api_key
        self.openai_api_key = open_ai_api_key
        self.model_names = model_names
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = Groq(api_key=self.api_key)

    def evaluate_dataset(self, dataset_path):
        global avg_faithfulness
        dataset = pd.read_csv(dataset_path, engine='python', encoding='utf-8', encoding_errors='ignore')
        model_results = {}

        for model_name in self.model_names:
            faithfulness_scores = []
            accuracy_scores = []
            relevance_scores = []
            faithfulness_evidences = []
            accuracy_evidences = []
            relevance_evidences = []

            for _, row in dataset.iterrows():
                question = row['question']
                context = row['contexts']
                answer = row['answer']
                ground_truth = row['ground_truths']

                print(f"question: {question}")

                #  groundedness_score = self.evaluate_groundedness(question, context, answer, model_name)
                accuracy_score, supporting_evidence_accuracy = self.evaluate_with_retry(self.evaluate_accuracy,
                                                                                        ground_truth, context, answer,
                                                                                        model_name)

                relevance_score, supporting_evidence_relevance = self.evaluate_with_retry(
                    self.evaluate_context_relevance,
                    question, context,
                    model_name)

                faithfulness_score, supporting_evidence_faithfulness = self.evaluate_with_retry(
                    self.evaluate_faithfulness, context,
                    answer, model_name)

                if accuracy_score is not None:
                    accuracy_scores.append(accuracy_score)
                    accuracy_evidences.append(supporting_evidence_accuracy)

                if relevance_score is not None:
                    relevance_scores.append(relevance_score)
                    relevance_evidences.append(supporting_evidence_relevance)

                if faithfulness_score is not None:
                    faithfulness_scores.append(faithfulness_score)
                    faithfulness_evidences.append(supporting_evidence_faithfulness)

                print(f"accuracy_scores {accuracy_scores}")
                print(f"relevance_scores: {relevance_scores}")
                print(f"faithfulness_scores {faithfulness_scores}")
                print("====================")

            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            if len(faithfulness_scores) > 0:
                avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
            else:
                avg_faithfulness = -1

            model_results[model_name] = {
                'faithfulness_scores': faithfulness_scores,
                'accuracy_scores': accuracy_scores,
                'relevance_scores': relevance_scores,
                'faithfulness_evidences': faithfulness_evidences,
                'accuracy_evidences': accuracy_evidences,
                'relevance_evidences': relevance_evidences,
                'avg_faithfulness': avg_faithfulness,
                'avg_accuracy': avg_accuracy,
                'avg_relevance': avg_relevance

            }

            self.create_graphs(model_results)
            self.save_results_to_csv(dataset, model_results)
        return model_results

    def evaluate_faithfulness(self, context, answer, model_name):
        faithfulness_prompt = ("\n\nContext: {context}\nAnswer:{answer}\nFaithfulness Score "
                               "and Reasons:")
        input_text = faithfulness_prompt.format(context=context, answer=answer)
        if model_name == 'gpt-3.5-turbo':
            self.client = OpenAI(api_key=self.openai_api_key)
        if model_name == 'claude-3-opus-20240229':
            self.client = Anthropic(api_key=self.anthropic_key)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompts.faithfulness_prompt_zero_shot
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=model_name,
            temperature=0.5,
        )
        return self.parse_score(model_name, chat_completion.choices[0])

    def evaluate_accuracy(self, ground_truth, context, answer, model_name):
        accuracy_prompt = ("\n\nAnswer: {answer}\nContext: {context}\n Ground Truth:{ground_truth}\n\nAccuracy Score "
                           "and Reason:")
        input_text = accuracy_prompt.format(ground_truth=ground_truth, context=context, answer=answer)
        if model_name == 'gpt-3.5-turbo':
            self.client = OpenAI(api_key=self.openai_api_key)
        if model_name == 'claude-3-opus-20240229':
            self.client = Anthropic(api_key=self.anthropic_key)

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompts.accuracy_prompt_zero_shot
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=model_name,
            temperature=0.5,
        )
        return self.parse_score(model_name, chat_completion.choices[0])

    def evaluate_context_relevance(self, question, context, model_name):

        user_prompt = " \n\nQuestion: {question}\nContext: {context}\n\n Context Relevance Score:"
        input_text = user_prompt.format(question=question, context=context)
        if model_name == 'gpt-3.5-turbo':
            self.client = OpenAI(api_key=self.openai_api_key)
        if model_name == 'claude-3-opus-20240229':
            self.client = Anthropic(api_key=self.anthropic_key)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": prompts.relevance_prompt_zero_shot
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ],
            model=model_name,
            temperature=0.5,

        )
        return self.parse_score(model_name, chat_completion.choices[0])

    def evaluate_with_retry(self, evaluate_func, *args):
        retries = 0
        while retries < self.max_retries:
            try:
                return evaluate_func(*args)
            except Exception:
                retries += 1
                print(Exception)
                print(f"Rate limit exceeded, retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        raise RuntimeError("Maximum retries exceeded for evaluation function.")

    def parse_score(self, model, response):
        output_text = response.message.content
        print(output_text)
        supporting_evidence_pattern = r'Supporting Evidence:\s*(.*)'
        score_pattern = r'Score:\s*(\d+\.\d+|\d+)'

        supporting_evidence_match = re.search(supporting_evidence_pattern, output_text, re.DOTALL)
        score_match = re.search(score_pattern, output_text)

        supporting_evidence = supporting_evidence_match.group(1).strip() if supporting_evidence_match else None
        score = float(score_match.group(1)) if score_match else None
        # print(score, model)
        return score, supporting_evidence

    def save_results_to_csv(self, dataset, model_results):
        for model_name, results in model_results.items():
            output_data = []
            for i, row in dataset.iterrows():
                question = row['question']
                answer = row['answer']
                context = row['contexts']
                ground_truth = row['ground_truths']
                context_relevance = results['relevance_scores'][i]
                context_relevance_evidence = results['relevance_evidences'][i]
                accuracy = results['accuracy_scores'][i]
                accuracy_evidence = results['accuracy_evidences'][i]
                faithfulness = results['faithfulness_scores'][i]
                faithfulness_evidence = results['faithfulness_evidences'][i]
                output_data.append(
                    [question, answer, context, ground_truth, context_relevance, context_relevance_evidence, accuracy,
                     accuracy_evidence, faithfulness, faithfulness_evidence])

            output_df = pd.DataFrame(output_data,
                                     columns=['question', 'answer', 'context', 'ground_truth', 'context_relevance',
                                              'context_relevance_evidence', 'accuracy', 'accuracy_evidence',
                                              'faithfulness', 'faithfulness_evidence'])
            output_file = f'results_{model_name}.csv'
            output_df.to_csv(output_file, index=False)

    def create_graphs(self, model_results):
        # Create a graph for groundedness
        plt.figure(figsize=(10, 6))
        groundedness_scores = [
            sum(model_results[model]['faithfulness_scores']) / len(model_results[model]['faithfulness_scores']) for
            model in model_results]
        plt.bar(model_results.keys(), groundedness_scores)
        plt.title('Average Faithfulness Scores')
        plt.xlabel('Model')
        plt.ylabel('Faithfulness Score')
        plt.savefig('faithfulness_scores.png')

        # Create a graph for accuracy
        plt.figure(figsize=(10, 6))
        accuracy_scores = [sum(model_results[model]['accuracy_scores']) / len(model_results[model]['accuracy_scores'])
                           for model in model_results]
        plt.bar(model_results.keys(), accuracy_scores)
        plt.title('Average Accuracy Scores')
        plt.xlabel('Model')
        plt.ylabel('Accuracy Score')
        plt.savefig('accuracy_scores.png')

        # Create a graph for context relevance
        plt.figure(figsize=(10, 6))
        relevance_scores = [
            sum(model_results[model]['relevance_scores']) / len(model_results[model]['relevance_scores']) for model in
            model_results]
        plt.bar(model_results.keys(), relevance_scores)
        plt.title('Average Context Relevance Scores')
        plt.xlabel('Model')
        plt.ylabel('Relevance Score')
        plt.savefig('relevance_scores.png')


if __name__ == "__main__":
    api_key =
    open_ai_api_key =
    anthropic_key = ()
    model_names = ["mixtral-8x7b-32768", "llama3-70b-8192", "gpt-3.5-turbo"]  # Add more model names as needed
    evaluator = LLMEvaluator(anthropic_key, api_key, open_ai_api_key, model_names)
    dataset_path = "D:/ENPM808/RAGRate/fiqa_eval_df.csv"
    model_results = evaluator.evaluate_dataset(dataset_path)

    for model, results in model_results.items():
        print(f"Model: {model}")
        print(f"Average Faithfulness: {results['avg_faithfulness']}")
        print(f"Average Accuracy: {results['avg_accuracy']}")
        print(f"Average Context Relevance: {results['avg_relevance']}")
        print()
