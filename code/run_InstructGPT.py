# evaluating GPT-3.5 turbo model on BBH

import openai
import json
import os
import numpy as np
import gpt3_tokenizer

from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
import logging

# parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str, default='key-series')
parser.add_argument('--model_index', type=str, default='text-ada-001', choices=['gpt-3.5', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4', 'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-001', 'text-davinci-002'])
parser.add_argument('--task', type=str, default='all', choices=['multiple_choice', 'free_form'])
parser.add_argument('--prompt', type=str, default='vanilla', choices=['vanilla', 'zero_shot_cot', 'manual_cot', 'PS', 'tot', 'self_consistency', 'auto_cot_pp', 'cotp'])
args = parser.parse_args()

MULTIPLE_CHOICE_TASKS = [
        'temporal_sequences',  'date_understanding', 'tracking_shuffled_objects_three_objects', 'penguins_in_a_table',
         'snarks', 'ruin_names', 'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects',
        'logical_deduction_three_objects', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_seven_objects', 'movie_recommendation',
         'reasoning_about_colored_objects', 'salient_translation_error_detection', 'disambiguation_qa', 'geometric_shapes',
]
FREE_FORM_TASKS = [
        'multistep_arithmetic_two', 'navigate',  'word_sorting', 'sports_understanding',
        'boolean_expressions', 'object_counting', 'causal_judgement', 'web_of_lies', 'dyck_languages','formal_fallacies',
]

@retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                       [wait_fixed(5) for i in range(2)] +
                       [wait_fixed(10)]))
def completion_with_backoff_instruct(**kwargs):
    return openai.Completion.create(**kwargs)

model_max_tokens = {
        'text-ada-001': 2048,
        'text-babbage-001': 2048,
        'text-curie-001': 2048,
        'text-davinci-001': 2048,
        'text-davinci-002': 4096,
    }
def extract_ans(ans, mode):
    ans_line = ans.split('answer is ')
    # Expect to see 'answer is'. If not return whole string
    if len(ans_line) == 1:
        return ans
    else:
        ans = ans_line[-1].strip()
    
    if mode == 'multiple_choice':
        options = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)', '(J)', '(K)', '(L)', '(M)', '(N)', '(O)', '(P)', '(Q)', '(R)', '(S)', '(T)', '(U)', '(V)', '(W)', '(X)', '(Y)', '(Z)']
        for option in options:
            if option in ans:
                ans = option[1]
                break
        return ans
    elif mode == 'free_form':
        if ans[-1] == '.':
            ans = ans[:-1]
        return ans

# Configure logging settings
def configure_logging(mode, prompt, model_index):
    log_path = f"outputs/log/{mode}/{prompt}/{model_index}.log"
    if not os.path.exists(f"outputs/log/{mode}/{prompt}/"):
        os.makedirs(f"outputs/log/{mode}/{prompt}/")
    logging.basicConfig(filename=log_path, level=logging.INFO)

def run_combined(tasks, mode, prompt, model_index="gpt-3.5-turbo"):
    assert prompt in ['vanilla', 'zero_shot_cot', 'manual_cot', 'PS', 'tot', 'self_consistency', 'auto_cot', 'cotp'], "Invalid run type"
    assert model_index in ['gpt-3.5-turbo', 'gpt-4', 'text-ada-001', 'text-babbage-001', 'text-curie-001', 'text-davinci-001', 'text-davinci-002'], "Invalid model index"
    configure_logging(mode, prompt, model_index)

    for task in tasks:
        logging.info('Testing %s ...', task)
        print('Testing %s ...' % task)
        acc = 0
        task_data = json.load(open('data/%s.json' % task))
        # Decide on the prompt based on the run type
        if prompt in ['vanilla', 'zero_shot_cot', 'manual_cot', 'self_consistency', 'PS', 'tot', 'auto_cot', 'cotp']:
            if prompt in ['manual_cot', 'self_consistency']:
                task_prompt = open('cot/%s.txt' % task, 'r').read()
            elif prompt == 'auto_cot':
                task_prompt = open('auto_cot/%s.txt' % task, 'r').read()
            elif prompt == 'cotp':
                task_prompt = open('cotp/%s.txt' % task, 'r').read()
            output_path = f'outputs/log/{mode}/{prompt}/test_{model_index}_{task}.txt'
            failure_path = f'outputs/log/{mode}/{prompt}/failure_cases_test_{model_index}_{task}.txt'
            if not os.path.exists(f"outputs/log/{mode}/{prompt}/"):
                os.makedirs(f"outputs/log/{mode}/{prompt}/")
        else:
            raise NotImplementedError("Unknown prompt")

        print_first = True
        with open(output_path, 'w') as fd, open(failure_path, 'w') as failure_fd:
            for q_ in tqdm(task_data['examples']):
                q = '\n\nQ: ' + q_['input']

                if prompt == 'vanilla':
                    prompt_q = q
                elif prompt == 'zero_shot_cot':
                    prompt_q = q + "\nA: Let's think step by step."
                elif prompt == 'manual_cot' or prompt == 'self_consistency' or prompt == 'cotp':
                    prompt_q = task_prompt + q + "\nA: Let's think step by step."
                elif prompt == 'PS':
                    prompt_q = task_prompt + q
                # https://github.com/AGI-Edgerunners/Plan-and-Solve-Prompting/
                elif prompt == 'tot':
                    prompt_q = task_prompt + q
                # https://github.com/dave1010/tree-of-thought-prompting
                elif prompt == 'auto_cot':
                    prompt_q = task_prompt + q + "\nA: Let's think step by step."
                else:
                    raise NotImplementedError

                if print_first:
                    logging.info('First prompt: %s [END]', prompt_q)
                    print('First prompt: %s [END]' % prompt_q)
                    print_first = False


                prompt_q = "Follow the given examples and answer the question.\n" + prompt_q
                max_model_tokens = model_max_tokens[model_index]

                prompt_tokens = len(gpt3_tokenizer.encode(prompt_q))
                max_completion_tokens = max_model_tokens - prompt_tokens
                response = completion_with_backoff_instruct(
                    model=model_index,
                    prompt=prompt_q,
                    temperature=0,
                    max_tokens=max_completion_tokens,
                )
                ans_model = response.choices[0].text.strip()
                ans_ = extract_ans(ans_model, mode)

                if mode == 'multiple_choice':
                    a = q_['target'][1]
                elif mode == 'free_form':
                    a = q_['target']
                fd.write('%s\nA_model:\n%s\nA_target:\n%s\n\n' % (q, ans_model, a))

                if ans_ == a:
                    acc += 1
                    # add True into log
                    logging.info('True')
                else:
                    # add False into log
                    logging.info('False')
                    # write the failure case, (the model answer ans_ and the target answer a) into
                    failure_fd.write('\n%s\nA_model:\n%s\nA_target:\n%s\n\n' % (q, ans_model, a))
                    # add ans_ too
                    failure_fd.write('extract_ans(A_model):\n%s\n\n' % (ans_))
            logging.info('%s acc %.4f', task, acc / len(task_data['examples']))
            print('%s acc %.4f' % (task, acc / len(task_data['examples'])))


def main(args, multiple_choice_tasks=MULTIPLE_CHOICE_TASKS, free_form_tasks=FREE_FORM_TASKS):
    # show args read easy to read
    print('args:', args)
    openai.api_key = args.api_key
    model_index = args.model_index
    run_multiple_choice = args.task == 'all' or args.task == 'multiple_choice'
    run_free_form = args.task == 'all' or args.task == 'free_form'
    if run_multiple_choice:
        run_combined(multiple_choice_tasks, prompt=args.prompt, mode='multiple_choice', model_index=model_index)
    if run_free_form:
        run_combined(free_form_tasks, prompt=args.prompt, mode='free_form', model_index=model_index)
    return 

if __name__ == '__main__':
    main(args)