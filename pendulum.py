import pybullet as p
import numpy as np

dt = 1 / 240  # pybullet simulation step
q0 = 0.5  # starting position (radian)
jIdx = 1
maxTime = 10
logTime = np.arange(0.0, maxTime, dt)
sz = logTime.size
logPos = np.zeros(sz)
logPos[0] = q0
idx = 0

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setGravity(0, 0, -10)
boxId = p.loadURDF("./pendulum.urdf", useFixedBase=True)

# Question:	The velocity of objects seems to be smaller than expected. Does PyBullet apply
# some default damping? Also the velocity doesn't exceed 100 units.
# Answer:	Yes, PyBullet applies some angular and linear damping to increase stability. You
# 		can modify/disable this damping using the 'changeDynamics' command, using
# 		linearDamping=0 and angularDamping=0 as arguments.
# The maximum linear/angular velocity is clamped to 100 units for stability.
# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit
# страница 87
# Это убрало затухания маятника
p.changeDynamics(boxId, jIdx, linearDamping=0, angularDamping=0)

# go to the starting position
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetPosition=q0, controlMode=p.POSITION_CONTROL)
for _ in range(1000):
    p.stepSimulation()

# turn off the motor for the free motion
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=jIdx, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)
p.setRealTimeSimulation(1)
for t in logTime[1:]:
    p.stepSimulation()
    # time.sleep(dt)

    jointState = p.getJointState(boxId, jIdx)
    th1 = jointState[0]
    idx += 1
    logPos[idx] = th1

import matplotlib.pyplot as plt

plt.grid(True)
plt.plot(logTime, logPos, label = "simPos")
plt.legend()

plt.show()

p.disconnect()


# Код из model.py, тут только для подсчёта L2 и Linf норм
maxTime = 10
L = 1.2
g = 10
logTime = np.arange(0.0, maxTime, dt)


def rp(x):
    return [x[1],
            -g / L * np.sin(x[0])]


def symplectic_euler(func, x0, t):
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        q_prev, p_prev = x[i - 1]
        p_next = p_prev - h * g / L * np.sin(q_prev)
        q_next = q_prev + h * p_next
        x[i] = [q_next, p_next]
    return x


theta = symplectic_euler(rp, [q0, 0], logTime)
logTheta = theta[:, 0]

print("L2 норма между logTheta и logPos:", np.sqrt(np.sum((logTheta - logPos) ** 2)))

print("Linf норма между logTheta и logPos:", np.max(np.abs(np.array(logTheta) - np.array(logPos))))
