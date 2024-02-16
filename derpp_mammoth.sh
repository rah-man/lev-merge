python main.py --model derpp_mod --dataset mod-cub200 --lr 3e-2 --alpha .1 --beta .5 --batch_size 10 --backbone_type whole --n_epochs 100 --ntask 10 --buffer_size 1000 --nowand 1 > ../output/derpp_cub200_10_1000_1.txt
python main.py --model derpp_mod --dataset mod-cub200 --lr 3e-2 --alpha .1 --beta .5 --batch_size 10 --backbone_type whole --n_epochs 100 --ntask 10 --buffer_size 2000 --nowand 1 > ../output/derpp_cub200_10_2000_1.txt
python main.py --model derpp_mod --dataset mod-cub200 --lr 3e-2 --alpha .1 --beta .5 --batch_size 10 --backbone_type whole --n_epochs 100 --ntask 20 --buffer_size 1000 --nowand 1 > ../output/derpp_cub200_20_1000_1.txt
python main.py --model derpp_mod --dataset mod-cub200 --lr 3e-2 --alpha .1 --beta .5 --batch_size 10 --backbone_type whole --n_epochs 100 --ntask 20 --buffer_size 2000 --nowand 1 > ../output/derpp_cub200_20_2000_1.txt