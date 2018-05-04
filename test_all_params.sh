python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --name baseline_001

python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --clip True --name clip_001
python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --dropout True --name dropout_001
python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --gradclip True --name gradclip_001

python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --clip True --dropout True --name clip_dropout_001
python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --clip True --gradclip True --name clip_gradclip_001
python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --dropout True --gradclip True --name dropout_gradclip_001

python driver.py --batch-size 100 --lr .001 --optimizer adam --seed 777 --epochs 15 --clip True --dropout True --gradclip True --name clip_dropout_gradclip_001

python driver.py --batch-size 512 --lr .0005 --optimizer adam --seed 777 --epochs 15  --model big --transform True --clip True --name transform_clip_0005_big
