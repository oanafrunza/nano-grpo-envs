#!/bin/bash
# Master launch script for Llama cross-validation experiments
# This script helps launch the vLLM servers and training jobs

set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       Llama Cross-Validation Experiment Launcher                    ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

show_usage() {
    echo "Usage: $0 <command> [model]"
    echo ""
    echo "Commands:"
    echo "  launch-vllm <3b|8b|all>   - Launch vLLM server(s)"
    echo "  launch-train <3b|8b|all>  - Launch training job(s)"
    echo "  status                    - Show running jobs"
    echo "  check-server <3b|8b>      - Check vLLM server health"
    echo ""
    echo "Examples:"
    echo "  $0 launch-vllm all        # Launch both vLLM servers"
    echo "  $0 launch-train 3b        # Launch 3B training"
    echo "  $0 status                 # Check all running jobs"
    echo ""
}

check_server() {
    local model=$1
    local port=$2
    
    echo "Checking vLLM server for ${model} on port ${port}..."
    if curl -s http://127.0.0.1:${port}/health > /dev/null 2>&1; then
        echo "✓ ${model} vLLM server is healthy on port ${port}"
        return 0
    else
        echo "✗ ${model} vLLM server is NOT reachable on port ${port}"
        return 1
    fi
}

launch_vllm() {
    local model=$1
    
    case $model in
        3b)
            echo "Launching Llama-3.2-3B vLLM server..."
            cd llama/llama_3b
            JOBID=$(sbatch sbatch_vllm_server.sh | awk '{print $4}')
            echo "✓ Submitted job ${JOBID} for Llama-3B vLLM (port 8001)"
            echo "  Monitor: tail -f logs/vllm_llama3b_${JOBID}.out"
            cd ../..
            ;;
        8b)
            echo "Launching Llama-3.1-8B vLLM server..."
            cd llama/llama_8b
            JOBID=$(sbatch sbatch_vllm_server.sh | awk '{print $4}')
            echo "✓ Submitted job ${JOBID} for Llama-8B vLLM (port 8002)"
            echo "  Monitor: tail -f logs/vllm_llama8b_${JOBID}.out"
            cd ../..
            ;;
        all)
            launch_vllm 3b
            echo ""
            launch_vllm 8b
            ;;
        *)
            echo "Error: Invalid model '$model'"
            echo "Use: 3b, 8b, or all"
            exit 1
            ;;
    esac
}

launch_training() {
    local model=$1
    
    case $model in
        3b)
            echo "Finding Llama-3B vLLM server node..."
            VLLM_NODE=$(squeue -u $USER -n vllm_llama3b -h -o "%N" | head -1)
            
            if [ -z "$VLLM_NODE" ]; then
                echo "✗ No Llama-3B vLLM server found!"
                echo "  Launch it first: $0 launch-vllm 3b"
                exit 1
            fi
            
            echo "✓ Found vLLM server on node: $VLLM_NODE"
            echo "Launching Llama-3B training..."
            cd llama/llama_3b
            JOBID=$(sbatch --nodelist=$VLLM_NODE sbatch_training.sh | awk '{print $4}')
            echo "✓ Submitted job ${JOBID} for Llama-3B training"
            echo "  Monitor: tail -f logs/llama3b_training_${JOBID}.out"
            cd ../..
            ;;
        8b)
            echo "Finding Llama-8B vLLM server node..."
            VLLM_NODE=$(squeue -u $USER -n vllm_llama8b -h -o "%N" | head -1)
            
            if [ -z "$VLLM_NODE" ]; then
                echo "✗ No Llama-8B vLLM server found!"
                echo "  Launch it first: $0 launch-vllm 8b"
                exit 1
            fi
            
            echo "✓ Found vLLM server on node: $VLLM_NODE"
            echo "Launching Llama-8B training..."
            cd llama/llama_8b
            JOBID=$(sbatch --nodelist=$VLLM_NODE sbatch_training.sh | awk '{print $4}')
            echo "✓ Submitted job ${JOBID} for Llama-8B training"
            echo "  Monitor: tail -f logs/llama8b_training_${JOBID}.out"
            cd ../..
            ;;
        all)
            launch_training 3b
            echo ""
            launch_training 8b
            ;;
        *)
            echo "Error: Invalid model '$model'"
            echo "Use: 3b, 8b, or all"
            exit 1
            ;;
    esac
}

show_status() {
    echo "Current Llama Experiment Jobs:"
    echo "─────────────────────────────────────────────────────────────────────"
    squeue -u $USER -n vllm_llama3b,vllm_llama8b,llama3b_training,llama8b_training -o "%.18i %.12j %.8T %.10M %.6D %R" || echo "No jobs running"
    echo ""
    
    echo "Checking vLLM Server Health:"
    echo "─────────────────────────────────────────────────────────────────────"
    check_server "Llama-3B" 8001 || true
    check_server "Llama-8B" 8002 || true
    echo ""
    
    echo "Experiment Progress:"
    echo "─────────────────────────────────────────────────────────────────────"
    
    # Check 3B experiments
    echo "Llama-3B:"
    for exp in baseline_seed0 continuous_fullzero_seed0 phase_adapt_seed0; do
        if [ -f "exp_output/llama_3b/${exp}.completed" ]; then
            echo "  ✓ ${exp}"
        else
            echo "  ⧖ ${exp}"
        fi
    done
    
    # Check 8B experiments
    echo "Llama-8B:"
    for exp in baseline_seed0 continuous_fullzero_seed0 phase_adapt_seed0; do
        if [ -f "exp_output/llama_8b/${exp}.completed" ]; then
            echo "  ✓ ${exp}"
        else
            echo "  ⧖ ${exp}"
        fi
    done
}

# Main command handler
case ${1:-""} in
    launch-vllm)
        if [ -z "$2" ]; then
            echo "Error: Specify model (3b, 8b, or all)"
            show_usage
            exit 1
        fi
        launch_vllm $2
        ;;
    launch-train)
        if [ -z "$2" ]; then
            echo "Error: Specify model (3b, 8b, or all)"
            show_usage
            exit 1
        fi
        launch_training $2
        ;;
    status)
        show_status
        ;;
    check-server)
        if [ -z "$2" ]; then
            echo "Error: Specify model (3b or 8b)"
            show_usage
            exit 1
        fi
        case $2 in
            3b)
                check_server "Llama-3B" 8001
                ;;
            8b)
                check_server "Llama-8B" 8002
                ;;
            *)
                echo "Error: Invalid model. Use 3b or 8b"
                exit 1
                ;;
        esac
        ;;
    help|--help|-h|"")
        show_usage
        ;;
    *)
        echo "Error: Unknown command '$1'"
        show_usage
        exit 1
        ;;
esac
