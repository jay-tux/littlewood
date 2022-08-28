#ifndef CAMERA_H
#define CAMERA_H

#include "glad/include/glad/glad.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifndef MIN_ZOOM
#define MIN_ZOOM 1.0f
#endif

#ifndef MAX_ZOOM
#define MAX_ZOOM 90.0f
#endif

typedef enum _direction { UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD } Direction;

class Camera {
    public:
        Camera();
        glm::vec3 *getCams();
        void moveCam(Direction d, float delta);
        void rotateCam(float, float);
        void zoomCam(float);
        void scrSizeCam(int, int);
        void update(glm::mat4 *, glm::mat4 *);

    private:
        bool firstm = true;
        float lastX, lastY, zoom = 45.0f, pitch = 0.0f, yaw = -90.0f;
        float sens = 0.1f, spd = 1.2f, aspect;
        glm::vec3 cam[3];
};

#endif
