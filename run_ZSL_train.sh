# python main.py --gpu_id 0 --exp_name Zsl_knowledge_space --exp_id W2V --fusion_model SAN --data_choice 3 --method_choice W2V --ZSL 1 --save_model 1
python main.py --gpu_id 0 --exp_name Zsl_semantic_space --exp_id W2V --fusion_model LSTMEncoder --data_choice 3 --method_choice W2V --ZSL 1 --save_model 1 --relation_map 1
python main.py --gpu_id 0 --exp_name Zsl_object_space --exp_id W2V --fusion_model SAN --data_choice 3 --method_choice W2V --ZSL 1 --save_model 1 --fact_map 1