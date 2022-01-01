from HAIPipeGen.HAIPipe import *

def generate_one_hai(notebook_id, haipipe_code_file='hai.py'):
    haipipe = HAIPipe(notebook_id)
    haipipe.evaluate_hi()
    haipipe.combine()
    haipipe.evalaute_hai()
    haipipe.output(haipipe_code_file)

if __name__ == "__main__":
    generate_one_hai('datascientist25_gender-recognition-by-voice-using-machine-learning')