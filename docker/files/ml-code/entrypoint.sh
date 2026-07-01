#!/bin/bash
###############################################################################
# v1 (uzairakbar/deepracer:v1) pure-simulation entrypoint.
#
# Rebuilds the behavior of the legacy uzairakbar/deepracer:v0 container on the
# modern deepracer-simapp stack (ROS 2 Jazzy / Gazebo Harmonic / Python 3.12).
# It runs the DeepRacer simulation as a pure, training-free environment exposed
# to P4_deepracer over a ZMQ REP socket (port ${GYM_PORT:-8888}), served by
# markov.gym_agent:GymAgent.
#
# Self-contained: an in-container MinIO provides the S3 endpoint the markov
# code expects, populated from the mounted /configs directory. No external
# trainer, no redis, no real AWS.
#
# Ported from P4_deepracer/patches/launch-simapp-rosnodes.sh.
###############################################################################
set -e
echo "Starting deepracer v1 pure-simulation entrypoint."

# --------------------------------------------------------------------------- #
# Runtime environment (mirror modern sageonly.sh / run.sh)
# --------------------------------------------------------------------------- #
export ROS_DISTRO=jazzy
export PYTHONUNBUFFERED=1
export XAUTHORITY=/root/.Xauthority
export TF_CPP_MIN_LOG_LEVEL=3
export DEEPRACER_JOB_TYPE_ENV="SAGEONLY"
export ROS_IP=127.0.0.1
export ENABLE_KINESIS=false
export ENABLE_GUI=false

# Per-user ROS 2 DDS isolation. Apptainer shares the host network namespace
# (Docker does not), so on a shared node (PACE) every user's nodes default to
# ROS_DOMAIN_ID=0 and discover each other over DDS -> cross-talk that corrupts
# topics/params. Give each user a distinct domain and restrict discovery to this
# host so we never reach nodes on other compute nodes. All processes in THIS
# container inherit these (exported before the launch), so intra-container comms
# are unaffected.
ros_domain_from_user() {
    local hash
    hash=$(echo -n "ROS_DOMAIN_${1}" | sha256sum | awk '{print $1}')
    # ROS 2 safe domain range is 1..101 (0 reserved as the shared default).
    echo $((1 + (16#${hash:0:8} % 101)))
}
export ROS_DOMAIN_ID=$(ros_domain_from_user "${USER:-deepracer}")
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID} (discovery range: ${ROS_AUTOMATIC_DISCOVERY_RANGE})"

export PATH="/opt/ml/:$PATH"
export PYTHONPATH="/opt/ml/code"

source /opt/ros/${ROS_DISTRO}/setup.bash
source /opt/amazon/install/setup.bash
source /root/anaconda/bin/activate sagemaker_env

# --------------------------------------------------------------------------- #
# World / object config from the mounted /configs (yq)
# --------------------------------------------------------------------------- #
echo "----------"
echo "EVALUATION: ${EVALUATION}"
echo "EVAL_WORLD_NAME: ${EVAL_WORLD_NAME}"
echo "GYM_PORT: ${GYM_PORT:-8888}"
echo "----------"

if [ "${EVALUATION}" = 'true' ] && [ -n "${EVAL_WORLD_NAME}" ]; then
    WORLD_NAME="${EVAL_WORLD_NAME}"
    echo "Running evaluation mode with ${WORLD_NAME} track."
else
    WORLD_NAME=$(yq .WORLD_NAME /configs/environment_params.yaml)
    echo "Running simulation mode with ${WORLD_NAME} track."
fi
NUMBER_OF_OBSTACLES=$(yq .NUMBER_OF_OBSTACLES /configs/environment_params.yaml)
NUMBER_OF_BOT_CARS=$(yq .NUMBER_OF_BOT_CARS /configs/environment_params.yaml)

# --------------------------------------------------------------------------- #
# S3 identifiers (local MinIO bucket)
# --------------------------------------------------------------------------- #
export AWS_REGION=us-east-1
export APP_REGION=${AWS_REGION}
S3_BUCKET="bucket"
SM_JOBNAME="rlexp-deepracer-prefix"
S3_PREFIX="sagemaker-${SM_JOBNAME}"

# MinIO credentials / endpoint used by the markov boto S3 client
export MINIO_ROOT_USER=minioadmin
export MINIO_ROOT_PASSWORD=minioadmin
export AWS_ACCESS_KEY_ID=${MINIO_ROOT_USER}
export AWS_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
S3_DATA_DIR=/opt/ml/s3data

# Pick free, non-colliding MinIO ports. Apptainer shares the host network
# namespace (Docker does not), so any fixed port (e.g. 9000) -- or even a
# per-user hashed port -- can already be held by another user or an unrelated
# service on a shared node (PACE), and hashing offers no way to detect that.
# MinIO is purely container-internal (reached only via the derived
# S3_ENDPOINT_URL below), so it needs a *free* port, not a *predictable* one:
# ask the OS for two currently-free ports in a single bind so they can never
# collide with each other or with anything else, and never fall back to 9000.
free_ports=$(python3 - <<'PY'
import socket
socks = [socket.socket() for _ in range(2)]
for s in socks:
    s.bind(("127.0.0.1", 0))
print(" ".join(str(s.getsockname()[1]) for s in socks))
for s in socks:
    s.close()
PY
)
MINIO_PORT=${free_ports%% *}
MINIO_CONSOLE_PORT=${free_ports##* }
export S3_ENDPOINT_URL="http://127.0.0.1:${MINIO_PORT}"

# --------------------------------------------------------------------------- #
# Start in-container MinIO and create the bucket
# --------------------------------------------------------------------------- #
mkdir -p "${S3_DATA_DIR}"
echo "Starting MinIO at ${S3_ENDPOINT_URL} (console :${MINIO_CONSOLE_PORT}, data: ${S3_DATA_DIR})"
minio server "${S3_DATA_DIR}" --address ":${MINIO_PORT}" \
    --console-address ":${MINIO_CONSOLE_PORT}" > /opt/ml/minio.log 2>&1 &
# wait for MinIO to accept connections
for i in $(seq 1 30); do
    if aws --endpoint-url "${S3_ENDPOINT_URL}" s3 ls > /dev/null 2>&1; then
        break
    fi
    sleep 1
done
aws --endpoint-url "${S3_ENDPOINT_URL}" s3 mb "s3://${S3_BUCKET}" 2>/dev/null || true

# --------------------------------------------------------------------------- #
# Populate the bucket: model_metadata, reward function, training params yaml
# --------------------------------------------------------------------------- #
REWARD_FUNCTION_S3_KEY=${S3_PREFIX}/custom_reward_function.py
MODEL_METADATA_S3_KEY=${S3_PREFIX}/model/model_metadata.json

# model_metadata == agent_params.json (action space / sensors / network)
aws --endpoint-url "${S3_ENDPOINT_URL}" s3 cp \
    /configs/agent_params.json "s3://${S3_BUCKET}/${MODEL_METADATA_S3_KEY}"

# reward function: P4 computes reward client-side, so the container copy is a
# placeholder; use the mounted one if present.
REWARD_SRC=/configs/reward_function.py
if [ ! -f "${REWARD_SRC}" ]; then
    REWARD_SRC=/opt/ml/code/zmq_default_reward_function.py
    printf 'def reward_function(params):\n    return 1.0\n' > "${REWARD_SRC}"
fi
aws --endpoint-url "${S3_ENDPOINT_URL}" s3 cp \
    "${REWARD_SRC}" "s3://${S3_BUCKET}/${REWARD_FUNCTION_S3_KEY}"

# Build training_params.yaml (defaults merged with /configs overrides), ported
# from the legacy launch-simapp-rosnodes.sh.
DEFAULT_YAML=/opt/ml/code/default_training_params.yaml
{
echo "WORLD_NAME:                           \"${WORLD_NAME}\""
echo "SAGEMAKER_SHARED_S3_BUCKET:           \"${S3_BUCKET}\""
echo "SAGEMAKER_SHARED_S3_PREFIX:           \"${S3_PREFIX}\""
# in-container MinIO endpoint, read by markov via WorldConfig.get_param so the
# rollout worker (and its boto S3 clients) talk to local MinIO, not real AWS.
echo "S3_ENDPOINT_URL:                      \"${S3_ENDPOINT_URL}\""
echo "TRAINING_JOB_ARN:                     \"local-sim\""
echo "METRICS_S3_BUCKET:                    \"${S3_BUCKET}\""
echo "METRICS_S3_OBJECT_KEY:                \"${S3_PREFIX}/training_metrics.json\""
echo "SIMTRACE_S3_BUCKET:                   \"${S3_BUCKET}\""
echo "SIMTRACE_S3_PREFIX:                   \"${S3_PREFIX}/iteration-data/training\""
echo "AWS_REGION:                           \"${AWS_REGION}\""
echo "TARGET_REWARD_SCORE:                  \"None\""
echo "NUMBER_OF_EPISODES:                   \"0\""
echo "JOB_TYPE:                             \"TRAINING\""
echo "CHANGE_START_POSITION:                \"true\""
echo "ALTERNATE_DRIVING_DIRECTION:          \"true\""
echo "REWARD_FILE_S3_KEY:                   \"${REWARD_FUNCTION_S3_KEY}\""
echo "MODEL_METADATA_FILE_S3_KEY:           \"${MODEL_METADATA_S3_KEY}\""
echo "NUMBER_OF_OBSTACLES:                  \"${NUMBER_OF_OBSTACLES}\""
echo "IS_OBSTACLE_BOT_CAR:                  \"false\""
echo "RANDOMIZE_OBSTACLE_LOCATIONS:         \"true\""
echo "IS_LANE_CHANGE:                       \"false\""
echo "LOWER_LANE_CHANGE_TIME:               \"3.0\""
echo "UPPER_LANE_CHANGE_TIME:               \"5.0\""
echo "LANE_CHANGE_DISTANCE:                 \"1.0\""
echo "NUMBER_OF_BOT_CARS:                   \"${NUMBER_OF_BOT_CARS}\""
echo "MIN_DISTANCE_BETWEEN_BOT_CARS:        \"2.0\""
echo "RANDOMIZE_BOT_CAR_LOCATIONS:          \"true\""
echo "BOT_CAR_SPEED:                        \"0.2\""
echo "CAR_COLOR:                            \"Blue\""
echo "NUMBER_OF_RESETS:                     \"0\""
echo "RACE_TYPE:                            \"HEAD_TO_BOT\""
echo "ENABLE_DOMAIN_RANDOMIZATION:          \"false\""
echo "DISPLAY_NAME:                         \"racer\""
echo "REVERSE_DIR:                          \"false\""
echo "BODY_SHELL_TYPE:                      \"deepracer\""
echo "IS_CONTINUOUS:                        \"false\""
echo "LEADERBOARD_NAME:                     \"cs7642\""
echo "NUM_WORKERS:                          \"1\""
} > "${DEFAULT_YAML}"

S3_YAML_NAME="training_params.yaml"
SOURCE_YAML=/configs/environment_params.yaml
# merge defaults (file 0) with /configs overrides (file 1)
yq eval-all 'select(fileIndex == 0) * select(fileIndex == 1)' "${DEFAULT_YAML}" "${SOURCE_YAML}" \
    > "/opt/ml/code/${S3_YAML_NAME}"
aws --endpoint-url "${S3_ENDPOINT_URL}" s3 cp \
    "/opt/ml/code/${S3_YAML_NAME}" "s3://${S3_BUCKET}/${S3_PREFIX}/${S3_YAML_NAME}"

# --------------------------------------------------------------------------- #
# Environment for the modern launch chain
#   distributed_training.launch.py -> download_params_and_roslaunch_agent.py
#   -> rollout_rl_agent.launch.py -> markov.rollout_worker (GymAgent ZMQ server)
# --------------------------------------------------------------------------- #
export WORLD_NAME=${WORLD_NAME}
export MODEL_S3_BUCKET=${S3_BUCKET}
export MODEL_S3_PREFIX=${S3_PREFIX}
export SAGEMAKER_SHARED_S3_BUCKET=${S3_BUCKET}
export SAGEMAKER_SHARED_S3_PREFIX=${S3_PREFIX}
export S3_YAML_NAME=${S3_YAML_NAME}
export SIMULATION_LAUNCH_FILE='distributed_training.launch.py'
export GAZEBO_MODEL_PATH='/opt/amazon/install/deepracer_simulation_environment/share/deepracer_simulation_environment'
export GYM_PORT=${GYM_PORT:-8888}

# --------------------------------------------------------------------------- #
# Headless display + launch the simulation (no trainer, no redis)
# --------------------------------------------------------------------------- #
export DISPLAY=:0
Xvfb "${DISPLAY}" -ac -screen 0 1400x900x24 > /opt/ml/xvfb.log 2>&1 &
sleep 2

echo "Launching simulation: ros2 launch deepracer_simulation_environment ${SIMULATION_LAUNCH_FILE}"
exec ros2 launch deepracer_simulation_environment "${SIMULATION_LAUNCH_FILE}"
