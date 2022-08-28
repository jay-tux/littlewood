#include "camera.h"

Camera::Camera()
{
    this->cam[0] = glm::vec3(0.0f, 0.0f,  3.0f); //pos
    this->cam[1] = glm::vec3(0.0f, 0.0f, -1.0f); //forward
    this->cam[2] = glm::vec3(0.0f, 1.0f,  0.0f); //up
}

glm::vec3 *Camera::getCams() { return this->cam; }

void Camera::zoomCam(float y)
{
    zoom -= (float)y;
    if(zoom < MIN_ZOOM) zoom = MIN_ZOOM;
    if(zoom > MAX_ZOOM) zoom = MAX_ZOOM;
}

void Camera::rotateCam(float x, float y)
{
    if(firstm)
    {
        lastX = x; lastY = y;
        firstm = false;
    }

    float dx = x - lastX;
    float dy = lastY - y;
    lastX = x; lastY = y;
    dx *= sens; dy *= sens;
    yaw += dx; pitch += dy;
    if(pitch >  89.0f) pitch =  89.0f;
    if(pitch < -89.0f) pitch = -89.0f;

    cam[1] = glm::vec3(
        cos(glm::radians(yaw)) * cos(glm::radians(pitch)),
        sin(glm::radians(pitch)),
        sin(glm::radians(yaw)) * cos(glm::radians(pitch))
    );
}

void Camera::moveCam(Direction d, float delta)
{
    float camspeed = spd * delta;
    switch(d)
    {
        case FORWARD:
            cam[0] += camspeed * cam[1];
            break;

        case LEFT:
            cam[0] -= camspeed * glm::normalize(glm::cross(cam[1], cam[2]));
            break;

        case RIGHT:
            cam[0] += camspeed * glm::normalize(glm::cross(cam[1], cam[2]));
            break;

        case BACKWARD:
            cam[0] -= camspeed * cam[1];
            break;

        case UP:
            cam[0] += camspeed * cam[2];
            break;

        case DOWN:
            cam[0] -= camspeed * cam[2];
            break;
    }
}

void Camera::scrSizeCam(int w, int h)
{
    glViewport(0, 0, w, h);
    this->aspect = (float)w / h;
}

void Camera::update(glm::mat4 *view, glm::mat4 *projection)
{
    *view = glm::lookAt(cam[0], cam[0] + cam[1], cam[2]);
    *projection = glm::perspective(glm::radians(zoom), aspect, 0.1f, 100.0f);
}
