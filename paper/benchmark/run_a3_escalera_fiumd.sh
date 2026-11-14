#!/bin/bash
# A3, escalera de temperaturas en FiuMD: cuadrado latino de 4 corridas.
# Cada corrida es un batch heterogeneo con las 4 temperaturas a la vez, una por
# especie; entre las 4 corridas cada especie pasa por las 4 temperaturas.
# Replicas por (especie, T): cln 16, trp 8, bbl 8, lambda 4.
#
#   nohup conda run -n fiumd --no-capture-output bash benchmark/run_a3_escalera_fiumd.sh >> logs/a3_escalera_fiumd.log 2>&1 &
cd "$(dirname "$0")/.."
echo "# Inicio: $(date)"
i=0
for temps in cln:300,trp:340,bbl:380,lambda:420 \
             cln:340,trp:380,bbl:420,lambda:300 \
             cln:380,trp:420,bbl:300,lambda:340 \
             cln:420,trp:300,bbl:340,lambda:380; do
    i=$((i + 1))
    echo "=== escalera fiumd $i/4 | $temps === $(date +%H:%M:%S)"
    # Relanzable: saltea las corridas completas y continua las cortadas
    dir=results/sistemas_multiples/a3_escalera_fiumd_$i
    resume=""
    if [ -e $dir/COMPLETED ]; then echo "ya esta completa"; continue; fi
    if [ -e $dir/checkpoint.npz ]; then resume="--resume"; elif [ -d $dir ]; then rm -r $dir; fi
    python -u benchmark/heterogeneous_ensemble.py --engine fiumd \
        --composition cln:16,trp:8,bbl:8,lambda:4 --nnp lambda --precision float32 \
        --temps $temps --dt 1.0 --dts bbl:0.5,lambda:0.5 --gamma 1.0 \
        --steps 300000 --stride 1000 --save_every 100 --seed $i \
        --run_id a3_escalera_fiumd_$i $resume || echo "!!! fallo la corrida $i"
done
echo "# Fin: $(date)"
