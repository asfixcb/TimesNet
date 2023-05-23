# TimesNet

python -u MyTimesNet.py --task_name long_term_forecast  --is_training 1  --root_path ./dataset/electricity/  --data_path electricity.csv  --model_id ECL_96_96  --model MyTimesNet  --data custom  --features M  --seq_len 100  --label_len 50  --pred_len 3  --e_layers 2  --d_layers 1  --factor 3  --enc_in 21  --dec_in 21  --c_out 1  --d_model 128  --d_ff 256  --top_k 5  --des 'Exp'  --itr 1
