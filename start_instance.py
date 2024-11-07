import time
import boto3
from botocore.exceptions import ClientError

# Create an EC2 client
ec2 = boto3.client('ec2', region_name='us-east-1')

# Define the instance ID
instance_id = "i-0593e3cae74779e4d"

# Function to start the instance
def start_instance():
    try:
        response = ec2.start_instances(InstanceIds=[instance_id])
        print(f"Starting instance {instance_id}: {response}")
        return True
    except ClientError as e:
        if "InsufficientInstanceCapacity" in str(e):
            print("Insufficient capacity. Retrying in 10 seconds...")
            return False
        else:
            print(f"Failed to start instance: {e}")
            raise

# Loop to retry starting the instance
while True:
    success = start_instance()

    if success:
        print(f"Instance {instance_id} started successfully.")
        break
    else:
        time.sleep(10)  # Wait 10 seconds before retrying
