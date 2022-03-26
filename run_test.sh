cd src

python infer_martin.py \
    --device cpu \
    --outdir '/home/martin/data/xournalpp_ml/worddetectornn/2_output_data/new' \
    --weights-path '/home/martin/data/xournalpp_ml/worddetectornn/0_trained_model/weights' \
    --images-path '/home/martin/data/xournalpp_ml/worddetectornn/1_input_data/own'
