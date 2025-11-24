# client.py
import grpc
import robot_pb2
import robot_pb2_grpc

def run():
    target = "172.30.1.9:50051"  # 서버의 내부망 IP:포트
    channel = grpc.insecure_channel(target)
    stub = robot_pb2_grpc.RobotControllerStub(channel)

    # 예제: 목적지 (x=1.23, y=4.56) 로 이동 요청
    response = stub.Navigate(robot_pb2.NavigateRequest(x=1.23, y=4.56))
    print("응답:", response.success, response.message)

if __name__ == "__main__":
    run()
