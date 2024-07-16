# Source-Free Domain-Invariant Performance Prediction

To run source-free performance evaluation on digits, run: 

``python pred_source_free.py --model_name lenet --model_dir /PATH/TO/MODELS/ --data_dir /PATH/TO/DATA/ --loss_type cross_entropy --log_dir ./LOG_DIR/--batch_size 64 --dname digits``


Main calculations for the calibration - `utils/multivar_distr.py`

GradNorm calculation - `utils/utils.py` (get_grad_norm_pytorch)
