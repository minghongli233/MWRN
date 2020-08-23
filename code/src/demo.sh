#===================================================================== MWRN-L =======================================================================
#-------------MWRN_L_x2 train test
python main.py --model MWRN_L --scale 2  --save MWRN_L_x2  --epochs 1000 --reset --patch_size 96 --cudnn
python main.py --model MWRN_L --scale 2  --pre_train ../experiment/model/MWRN_L_x2.pt --save MWRN_L_x2 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_L --scale 2  --pre_train ../experiment/model/MWRN_L_x2.pt --save MWRN_L_plus_x2 --test_only --save_results --self_ensemble --data_test Set5

#-------------MWRN_L_x3 train test
python main.py --model MWRN_L --scale 3  --save MWRN_L_x3  --epochs 1000 --reset --patch_size 144 --cudnn
python main.py --model MWRN_L --scale 3  --pre_train ../experiment/model/MWRN_L_x3.pt --save MWRN_L_x3 --test_only --data_test Set5 --save_results
python main.py --model MWRN_L --scale 3  --pre_train ../experiment/model/MWRN_L_x3.pt --save MWRN_L_plus_x3 --test_only --save_results --self_ensemble --data_test Set5

#-------------MWRN_L_x4 train test
python main.py --model MWRN_L --scale 4  --save MWRN_L_x4  --epochs 1000 --reset --patch_size 192 --cudnn
python main.py --model MWRN_L --scale 4  --pre_train ../experiment/model/MWRN_L_x4.pt --save MWRN_L_x4 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_L --scale 4  --pre_train ../experiment/model/MWRN_L_x4.pt --save MWRN_L_plus_x4 --test_only --save_results --self_ensemble --data_test Set5 

#===================================================================== MWRN-M =======================================================================
#-------------MWRN_M_x2 train test
python main.py --model MWRN_M --scale 2  --save MWRN_M_x2  --epochs 1000 --reset --patch_size 96 --cudnn
python main.py --model MWRN_M --scale 2  --pre_train ../experiment/model/MWRN_M_x2.pt --save MWRN_M_x2 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_M --scale 2  --pre_train ../experiment/model/MWRN_M_x2.pt --save MWRN_M_plus_x2 --test_only  --save_results --self_ensemble --data_test Set5

#-------------MWRN_M_x3 train test
python main.py --model MWRN_M --scale 3  --save MWRN_M_x3  --epochs 1000 --reset --patch_size 144 --cudnn
python main.py --model MWRN_M --scale 3  --pre_train ../experiment/model/MWRN_M_x3.pt --save MWRN_M_x3 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_M --scale 2  --pre_train ../experiment/model/MWRN_M_x2.pt --save MWRN_M_plus_x2 --test_only  --save_results --self_ensemble --data_test Set5

#-------------MWRN_M_x4 train test
python main.py --model MWRN_M --scale 4  --save MWRN_M_x4  --epochs 1000 --reset --patch_size 192 --cudnn
python main.py --model MWRN_M --scale 4  --pre_train ../experiment/model/MWRN_M_x4.pt --save MWRN_M_x4 --test_only --data_test Set5 --save_results
python main.py --model MWRN_M --scale 2  --pre_train ../experiment/model/MWRN_M_x2.pt --save MWRN_M_plus_x2 --test_only  --save_results --self_ensemble --data_test Set5

#===================================================================== MWRN-H =======================================================================
#-------------MWRN_H_x2 train test
python main.py --model MWRN_H --scale 2  --save MWRN_H_x2  --epochs 1000 --reset --patch_size 96 --cudnn
python main.py --model MWRN_H --scale 2  --pre_train ../experiment/model/MWRN_H_x2.pt --save MWRN_H_x2 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_H --scale 2  --pre_train ../experiment/model/MWRN_H_x2.pt --save MWRN_H_plus_x2 --test_only  --save_results --self_ensemble --data_test Set5

#-------------MWRN_H_x3 train test
python main.py --model MWRN_H --scale 3  --save MWRN_H_x3  --epochs 1000 --reset --patch_size 144 --cudnn
python main.py --model MWRN_H --scale 3  --pre_train ../experiment/model/MWRN_H_x3.pt --save MWRN_H_x3 --test_only --data_test Set5 --save_results 
python main.py --model MWRN_H --scale 3  --pre_train ../experiment/model/MWRN_H_x3.pt --save MWRN_H_plus_x3 --test_only  --save_results --self_ensemble --data_test Set5

#-------------MWRN_H_x4 train test
python main.py --model MWRN_H --scale 4  --save MWRN_H_x4  --epochs 1000 --reset --patch_size 192 --cudnn
python main.py --model MWRN_H --scale 4  --pre_train ../experiment/model/MWRN_H_x4.pt --save MWRN_H_x4 --test_only --data_test Set5 --save_results
python main.py --model MWRN_H --scale 4  --pre_train ../experiment/model/MWRN_H_x4.pt --save MWRN_H_plus_x4 --test_only  --save_results --self_ensemble --data_test Set5



