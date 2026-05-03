using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System;

public class SimpleGridBalls : MonoBehaviour {
    [Header("小球预制体")]
    public GameObject ballPrefab;

    [Header("XZ 平面范围与步长")]
    public Vector2 xRange = new Vector2(-0.1494f, 0.1494f);
    public Vector2 zRange = new Vector2(-0.1182f, 0.1182f);
    public float step = 0.001f;

    [Header("起始Y高度及步进")]
    public float startY = 0.053f;
    public float yStep = 0.001f;
    public float maxY = 0.332f;

    [Header("速度方向（8个对角线方向）")]
    private Vector3[] directions = new Vector3[]
    {
        new Vector3( 1, 1, 1), new Vector3( 1, 1,-1),
        new Vector3( 1,-1, 1), new Vector3( 1,-1,-1),
        new Vector3(-1, 1, 1), new Vector3(-1, 1,-1),
        new Vector3(-1,-1, 1), new Vector3(-1,-1,-1)
    };
    public float speedMagnitude = 1f;

    [Header("数据保存")]
    public string dataFileName = "grid_collision_data";
    public int autoSaveInterval = 50;

    private class BallState {
        public BallMove ball;
        public float currentY;
        public int dirIndex;
        public Vector3 originalXZ;   // 固定的XZ坐标
    }
    private List<BallState> ballStates = new List<BallState>();

    private FileStream fs;
    private BinaryWriter writer;
    private int recordCount = 0;
    private int pendingCount = 0;
    private object lockObj = new object();

    void Start() {
        GenerateGridBalls();
        SetupDataFile();
        IgnoreBallCollisions();
    }

    void GenerateGridBalls() {
        int xCount = Mathf.FloorToInt((xRange.y - xRange.x) / step) + 1;
        int zCount = Mathf.FloorToInt((zRange.y - zRange.x) / step) + 1;
        Vector3 initialVelocity = directions[0].normalized * speedMagnitude;

        for (int i = 0; i < xCount; i++) {
            float x = xRange.x + i * step;
            for (int j = 0; j < zCount; j++) {
                float z = zRange.x + j * step;
                Vector3 pos = new Vector3(x, startY, z);

                GameObject go = Instantiate(ballPrefab, pos, Quaternion.identity);
                go.GetComponent<Renderer>().material.color = UnityEngine.Random.ColorHSV();
                BallMove ball = go.GetComponent<BallMove>();
                ball.autoCollectMode = true;
                ball.velocity = initialVelocity;

                BallState state = new BallState {
                    ball = ball,
                    currentY = startY,
                    dirIndex = 0,
                    originalXZ = new Vector3(x, 0, z)
                };
                ballStates.Add(state);

                // 订阅碰撞事件，使用局部变量捕获state
                ball.OnHitBoard += (hitPoint, ballPos, ballVel) => {
                    // 注意：这里 ballPos 和 ballVel 是碰撞前的瞬时值，我们不使用它们。
                    // 而是直接使用 state 中保存的起始参数。
                    OnBallCollision(state, hitPoint);
                };
            }
        }
        Debug.Log($"生成了 {ballStates.Count} 个小球");
    }

    void SetupDataFile() {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string path = Path.Combine(@"D:\SSAS\", $"{dataFileName}_{timestamp}.bin");
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        fs = new FileStream(path, FileMode.Create);
        writer = new BinaryWriter(fs);
        writer.Write((int)0x424C4430);
        writer.Write(0);   // 记录数占位
        fs.Flush();
    }

    void IgnoreBallCollisions() {
        for (int i = 0; i < ballStates.Count; i++) {
            Collider collA = ballStates[i].ball.GetComponent<Collider>();
            for (int j = i + 1; j < ballStates.Count; j++) {
                Collider collB = ballStates[j].ball.GetComponent<Collider>();
                Physics.IgnoreCollision(collA, collB, true);
            }
        }
    }

    void OnBallCollision(BallState state, Vector3 hitPoint) {
        // 本次试验的起始位置（重置时的位置）和起始速度
        Vector3 startPos = new Vector3(state.originalXZ.x, state.currentY, state.originalXZ.z);
        Vector3 startVel = directions[state.dirIndex].normalized * speedMagnitude;

        // 写入数据：起始位置，起始速度，碰撞点
        lock (lockObj) {
            writer.Write(startPos.x);
            writer.Write(startPos.y);
            writer.Write(startPos.z);
            writer.Write(startVel.x);
            writer.Write(startVel.y);
            writer.Write(startVel.z);
            writer.Write(hitPoint.x);
            writer.Write(hitPoint.y);
            writer.Write(hitPoint.z);
            recordCount++;
            pendingCount++;
        }

        if (pendingCount >= autoSaveInterval)
            FlushToDisk();

        // --- 更新状态，为下一次试验做准备 ---
        state.dirIndex++;
        if (state.dirIndex >= directions.Length) {
            state.dirIndex = 0;
            state.currentY += yStep;
        }

        // 检查是否完成所有组合
        if (state.currentY > maxY + 0.001f)   // 允许微小浮点误差
        {
            Destroy(state.ball.gameObject);
            ballStates.Remove(state);
            Debug.Log($"一个球完成，剩余 {ballStates.Count} 个球");
            return;
        }

        // 重置球到新的起始位置和速度
        Vector3 newPos = new Vector3(state.originalXZ.x, state.currentY, state.originalXZ.z);
        Vector3 newVel = directions[state.dirIndex].normalized * speedMagnitude;
        state.ball.ResetBall(newPos, newVel);
    }

    void FlushToDisk() {
        lock (lockObj) {
            if (pendingCount == 0) return;
            fs.Flush();
            long originalPos = fs.Position;
            fs.Seek(4, SeekOrigin.Begin);
            writer.Write(recordCount);
            fs.Seek(originalPos, SeekOrigin.Begin);
            fs.Flush();
            pendingCount = 0;
            Debug.Log($"已保存，总记录数: {recordCount}");
        }
    }

    void OnDestroy() {
        FlushToDisk();
        writer?.Close();
        fs?.Close();
    }
}