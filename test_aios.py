import aios
import numpy as np

def main():
    print("Connecting to AIOS daemon...")
    with aios.Client() as client:
        print(f"Connected over {client.transport} transport!")
        input_data = np.random.rand(1, 3, 64, 64).astype(np.float32)
        
        result = client.infer(
            "vision-cnn", 
            input_data, 
            deadline_ms=80, 
            objective="latency"
        )
        
        print(f"Target selected by AIOS: {result.target}")
        print(f"Execution Latency:       {result.latency_ms:.2f} ms")
        print(f"Predicted Class:         {result.argmax()[0]}")

if __name__ == "__main__":
    main()
