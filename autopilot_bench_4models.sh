#!/bin/bash
# Benchmark the four Autopilot detectors (YOLO-World, SAM3, Grounding DINO, OWLv2) and
# CLIP-large vs CLIP-base on the 18 hand-labeled videos. Nothing is downloaded and no
# VIAT project is touched; results go to reports/ and logs to reports/logs/.
#
#   ./autopilot_bench_4models.sh            # both GPUs in step 1 (faster)
#   ONE_GPU=1 ./autopilot_bench_4models.sh  # step 1 one model at a time (lighter load)
#
# Safe to stop (Ctrl+C) and run again: detections are cached and the run resumes.

cd "$(dirname "$(readlink -f "$0")")" || exit 1
mkdir -p reports/logs
LOG=reports/logs

D="$HOME/Videos/Done"
VIDEOS=(
  AchzaritTank 20_INCREDIBLE_ARMORED_VEHICLES BMW_X6_UavView CarsInSnowDroneView DroneViewHuman
  IDF_D9_armored_bulldozer Danish_troops_firing DifferentVanets Challenger3Tank
  "AN_MPQ-64 Sentinel RADAR - Andrew Dutton (720p, h264)_n_blurred" CV90_BrazilTank_train DroneRunChicago1
  ISrealEitanAPC1 INKAS_M1MilitaryVehicle Israeli_bulldozers_manoeuver ImaginiK2TankTrain BMW_M_DRAG EpicRunDroneView
)
V=()
for v in "${VIDEOS[@]}"; do
  if [ -f "$D/$v.mp4" ]; then V+=(--video "$D/$v.mp4"); else echo "skipping missing video: $D/$v.mp4"; fi
done

COMMON=(--frames 600 --quality high)
GDINO=IDEA-Research/grounding-dino-base
OWL=google/owlv2-large-patch14-ensemble
ALL=(--model yolov8x-worldv2.pt --model sam3.1_l.pt --model "$GDINO" --model "$OWL")

trap 'echo; echo "stopped - run the script again to resume"; kill 0' INT TERM

step() { echo "=== $(date '+%H:%M:%S') $*"; }

step "1/3 caching Grounding DINO + OWLv2 detections (longest step; progress: tail -f $LOG/fill_*.log)"
if [ -n "$ONE_GPU" ]; then
  python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" --model "$GDINO" --judge none \
    --report $LOG/fill_gdino.md > $LOG/fill_gdino.log 2>&1 || { echo "Grounding DINO failed, see $LOG/fill_gdino.log"; exit 1; }
  python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" --model "$OWL" --judge none \
    --report $LOG/fill_owl.md > $LOG/fill_owl.log 2>&1 || { echo "OWLv2 failed, see $LOG/fill_owl.log"; exit 1; }
else
  CUDA_VISIBLE_DEVICES=0 python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" --model "$GDINO" --judge none \
    --report $LOG/fill_gdino.md > $LOG/fill_gdino.log 2>&1 &
  P1=$!
  CUDA_VISIBLE_DEVICES=1 python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" --model "$OWL" --judge none \
    --report $LOG/fill_owl.md > $LOG/fill_owl.log 2>&1 &
  P2=$!
  wait $P1 || { echo "Grounding DINO failed, see $LOG/fill_gdino.log"; kill $P2 2>/dev/null; exit 1; }
  wait $P2 || { echo "OWLv2 failed, see $LOG/fill_owl.log"; exit 1; }
fi

step "2/3 four detectors alone and combined, with judges"
python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" "${ALL[@]}" \
  --judge clip --judge sam_zoom --judge learned --compare-models \
  --report reports/autopilot_benchmark_4models.md > $LOG/bench4.log 2>&1 \
  || { echo "step 2 failed, see $LOG/bench4.log"; exit 1; }

step "3/3 CLIP-large judge"
python -u autopilot_benchmark.py "${COMMON[@]}" "${V[@]}" "${ALL[@]}" \
  --judge clip --clip-model openai/clip-vit-large-patch14 \
  --report reports/autopilot_benchmark_cliplarge.md > $LOG/bench_cliplarge.log 2>&1 \
  || { echo "step 3 failed, see $LOG/bench_cliplarge.log"; exit 1; }

step "done"
echo "reports: reports/autopilot_benchmark_4models.md  reports/autopilot_benchmark_cliplarge.md"
