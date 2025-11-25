import grpc
import bero_pb2
import bero_pb2_grpc

def run():
    target = "100.76.44.116:50051"  # 서버 IP:포트
    channel = grpc.insecure_channel(target)
    stub = bero_pb2_grpc.RobotControllerStub(channel)

    # 1. 목적지 waypoint 이름으로 이동 요청
    waypoint_name = "corridor_center"  # 실제 waypoints.yaml에 있는 이름으로 변경 필요
    nav_response = stub.Navigate(bero_pb2.NavigateRequest(waypoint=waypoint_name))
    print("[Navigate] 성공:", nav_response.success, "| 메시지:", nav_response.message)

    # 2. 상황 설명 요청
    prompt = "복도에 사람이 있는지 알려줘"
    scene_response = stub.DescribeScene(bero_pb2.SceneRequest(prompt=prompt))
    print("[DescribeScene] 성공:", scene_response.success, "| 설명:", scene_response.description)

if __name__ == "__main__":
    run()