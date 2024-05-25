data=3;
ke=25;
kr=25;
score=100;
zsl=1;
python joint_test.py --gpu_id 0 --exp_name fusion_prediction_zsl --ZSL "${zsl}" --exp_id rel"${kr}"_fact"${ke}"data_"${data}"score_"${score}" --data_choice "${data}" --top_rel "${kr}" --top_fact "${ke}" --soft_score "${score}"  --mrr 1
