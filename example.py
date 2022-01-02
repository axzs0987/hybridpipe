from HybridPipeGen.HybridPipe import *

<<<<<<< HEAD
def generate_one_hybrid(notebook_path, dataset_path, label_index, model, hybridpipe_code_file='hybrid_example.py'):
    hybrid_pipe = HybridPipe(notebook_path, dataset_path, label_index, model)
    hybrid_pipe.evaluate_hi()
    hybrid_pipe.generate_mlpipe()
    hybrid_pipe.combine()
    hybrid_pipe.evalaute_hybrid()
    hybrid_pipe.output(hybridpipe_code_file,save_fig=True)

if __name__ == "__main__":
    notebook_path = 'data/notebook/datascientist25_gender-recognition-by-voice-using-machine-learning.ipynb'
    dataset_path = 'data/dataset/primaryobjects_voicegender/voice.csv'
    label_index = 20
    support_model = ['RandomForestClassifier', 'KNeighborsClassifier', 'LogisticRegression', 'SVC']
    generate_one_hybrid(notebook_path, dataset_path, label_index, support_model[2])
=======
def generate_one_hai(notebook_id, haipipe_code_file='hybrid_example.py'):
    haipipe = HybridPipe(notebook_id)
    haipipe.evaluate_hi()
    haipipe.combine()
    haipipe.evalaute_hai()
    haipipe.output(haipipe_code_file,save_fig=True)

if __name__ == "__main__":
    generate_one_hai('datascientist25_gender-recognition-by-voice-using-machine-learning')
>>>>>>> 667bbf9cb98902e69354ddf34ecd5389817d14f7
