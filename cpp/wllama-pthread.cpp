/**
 * Runtime pthread intercept layer.
 *
 * When support_pthread == false every pthread call is answered with a
 * single-threaded stub (mirrors Emscripten's library_pthread_stub.c).
 * When support_pthread == true every call is forwarded to the real
 * implementation via the __real_* symbols provided by --wrap linkage.
 */

#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static struct s_support_pthread {
    bool value = false;
    s_support_pthread() {
        const char *env = getenv("SUPPORT_PTHREAD");
        if (env && strcmp(env, "1") == 0) {
            printf("pthread is enabled\n");
            value = true;
        }
    }
    operator bool() const { return value; }
} support_pthread;

// ---------------------------------------------------------------------------
// Single-threaded TLS emulation (pthread_key / getspecific / setspecific)
// ---------------------------------------------------------------------------
#define STUB_MAX_KEYS 128
static void  *s_tls_values[STUB_MAX_KEYS];
static int    s_next_key = 0;

extern "C" {

// ---------------------------------------------------------------------------
// __real_* forward declarations
// ---------------------------------------------------------------------------

int   __real_pthread_mutex_init(pthread_mutex_t * __restrict, const pthread_mutexattr_t * __restrict);
int   __real___pthread_mutex_lock(pthread_mutex_t *);
int   __real___pthread_mutex_unlock(pthread_mutex_t *);
int   __real___pthread_mutex_trylock(pthread_mutex_t *);
int   __real___pthread_mutex_timedlock(pthread_mutex_t * __restrict, const struct timespec * __restrict);
int   __real_pthread_mutex_destroy(pthread_mutex_t *);
int   __real_pthread_mutex_consistent(pthread_mutex_t *);

int   __real_pthread_barrier_init(pthread_barrier_t * __restrict, const pthread_barrierattr_t * __restrict, unsigned);
int   __real_pthread_barrier_destroy(pthread_barrier_t *);
int   __real_pthread_barrier_wait(pthread_barrier_t *);

int   __real___pthread_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
int   __real___pthread_join(pthread_t, void **);

int   __real___pthread_key_create(pthread_key_t *, void (*)(void *));
int   __real___pthread_key_delete(pthread_key_t);
void *__real_pthread_getspecific(pthread_key_t);
int   __real_pthread_setspecific(pthread_key_t, const void *);

int   __real___pthread_once(pthread_once_t *, void (*)(void));

int   __real_pthread_cond_wait(pthread_cond_t * __restrict, pthread_mutex_t * __restrict);
int   __real_pthread_cond_signal(pthread_cond_t *);
int   __real_pthread_cond_broadcast(pthread_cond_t *);
int   __real_pthread_cond_init(pthread_cond_t * __restrict, const pthread_condattr_t * __restrict);
int   __real_pthread_cond_destroy(pthread_cond_t *);
int   __real___pthread_cond_timedwait(pthread_cond_t * __restrict, pthread_mutex_t * __restrict, const struct timespec * __restrict);

int   __real_pthread_atfork(void (*)(void), void (*)(void), void (*)(void));
int   __real_pthread_cancel(pthread_t);
void  __real___pthread_testcancel(void);
void  __real___pthread_exit(void *) __attribute__((noreturn));
int   __real___pthread_detach(pthread_t);
int   __real_pthread_equal(pthread_t, pthread_t);
int   __real_pthread_kill(pthread_t, int);
int   __real_pthread_setcancelstate(int, int *);
int   __real_pthread_setcanceltype(int, int *);

int   __real_pthread_rwlock_init(pthread_rwlock_t * __restrict, const pthread_rwlockattr_t * __restrict);
int   __real_pthread_rwlock_destroy(pthread_rwlock_t *);
int   __real_pthread_rwlock_rdlock(pthread_rwlock_t *);
int   __real_pthread_rwlock_tryrdlock(pthread_rwlock_t *);
int   __real_pthread_rwlock_timedrdlock(pthread_rwlock_t * __restrict, const struct timespec * __restrict);
int   __real_pthread_rwlock_wrlock(pthread_rwlock_t *);
int   __real_pthread_rwlock_trywrlock(pthread_rwlock_t *);
int   __real_pthread_rwlock_timedwrlock(pthread_rwlock_t * __restrict, const struct timespec * __restrict);
int   __real_pthread_rwlock_unlock(pthread_rwlock_t *);

int   __real_sem_post(sem_t *);
int   __real_sem_wait(sem_t *);
int   __real_sem_trywait(sem_t *);

// ---------------------------------------------------------------------------
// Mutex
// ---------------------------------------------------------------------------

int __wrap_pthread_mutex_init(pthread_mutex_t * __restrict m, const pthread_mutexattr_t * __restrict a) {
    if (support_pthread) return __real_pthread_mutex_init(m, a);
    return 0;
}
int __wrap___pthread_mutex_lock(pthread_mutex_t *m) {
    if (support_pthread) return __real___pthread_mutex_lock(m);
    return 0;
}
int __wrap___pthread_mutex_unlock(pthread_mutex_t *m) {
    if (support_pthread) return __real___pthread_mutex_unlock(m);
    return 0;
}
int __wrap___pthread_mutex_trylock(pthread_mutex_t *m) {
    if (support_pthread) return __real___pthread_mutex_trylock(m);
    return 0;
}
int __wrap___pthread_mutex_timedlock(pthread_mutex_t * __restrict m, const struct timespec * __restrict t) {
    if (support_pthread) return __real___pthread_mutex_timedlock(m, t);
    return 0;
}
int __wrap_pthread_mutex_destroy(pthread_mutex_t *m) {
    if (support_pthread) return __real_pthread_mutex_destroy(m);
    return 0;
}
int __wrap_pthread_mutex_consistent(pthread_mutex_t *m) {
    if (support_pthread) return __real_pthread_mutex_consistent(m);
    return 0;
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

int __wrap_pthread_barrier_init(pthread_barrier_t * __restrict b, const pthread_barrierattr_t * __restrict a, unsigned u) {
    if (support_pthread) return __real_pthread_barrier_init(b, a, u);
    return 0;
}
int __wrap_pthread_barrier_destroy(pthread_barrier_t *b) {
    if (support_pthread) return __real_pthread_barrier_destroy(b);
    return 0;
}
int __wrap_pthread_barrier_wait(pthread_barrier_t *b) {
    if (support_pthread) return __real_pthread_barrier_wait(b);
    return PTHREAD_BARRIER_SERIAL_THREAD;
}

// ---------------------------------------------------------------------------
// Thread lifecycle
// ---------------------------------------------------------------------------

int __wrap___pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    if (support_pthread) return __real___pthread_create(t, a, fn, arg);
    return EAGAIN;
}
int __wrap___pthread_join(pthread_t t, void **retval) {
    if (support_pthread) return __real___pthread_join(t, retval);
    return 0;
}
int __wrap___pthread_detach(pthread_t t) {
    if (support_pthread) return __real___pthread_detach(t);
    return 0;
}
int __wrap_pthread_cancel(pthread_t t) {
    if (support_pthread) return __real_pthread_cancel(t);
    return 0;
}
void __wrap___pthread_testcancel(void) {
    if (support_pthread) __real___pthread_testcancel();
}
void __wrap___pthread_exit(void *status) {
    if (support_pthread) __real___pthread_exit(status);
    exit(0);
}
int __wrap_pthread_equal(pthread_t t1, pthread_t t2) {
    if (support_pthread) return __real_pthread_equal(t1, t2);
    return t1 == t2;
}
int __wrap_pthread_kill(pthread_t t, int sig) {
    if (support_pthread) return __real_pthread_kill(t, sig);
    return ESRCH;
}
int __wrap_pthread_setcancelstate(int state, int *old) {
    if (support_pthread) return __real_pthread_setcancelstate(state, old);
    return 0;
}
int __wrap_pthread_setcanceltype(int type, int *old) {
    if (support_pthread) return __real_pthread_setcanceltype(type, old);
    return 0;
}

// ---------------------------------------------------------------------------
// Thread-local storage (key / getspecific / setspecific)
// ---------------------------------------------------------------------------

int __wrap___pthread_key_create(pthread_key_t *key, void (*dtor)(void *)) {
    if (support_pthread) return __real___pthread_key_create(key, dtor);
    if (s_next_key >= STUB_MAX_KEYS) return ENOMEM;
    *key = s_next_key++;
    return 0;
}
int __wrap___pthread_key_delete(pthread_key_t key) {
    if (support_pthread) return __real___pthread_key_delete(key);
    return 0;
}
void *__wrap_pthread_getspecific(pthread_key_t key) {
    if (support_pthread) return __real_pthread_getspecific(key);
    if ((int)key >= STUB_MAX_KEYS) return nullptr;
    return s_tls_values[key];
}
int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (support_pthread) return __real_pthread_setspecific(key, value);
    if ((int)key >= STUB_MAX_KEYS) return EINVAL;
    s_tls_values[key] = const_cast<void *>(value);
    return 0;
}

// ---------------------------------------------------------------------------
// Once
// ---------------------------------------------------------------------------

int __wrap___pthread_once(pthread_once_t *ctrl, void (*fn)(void)) {
    if (support_pthread) return __real___pthread_once(ctrl, fn);
    if (*ctrl) return 0;
    *ctrl = 1;
    fn();
    return 0;
}

// ---------------------------------------------------------------------------
// Condition variables
// ---------------------------------------------------------------------------

int __wrap_pthread_cond_init(pthread_cond_t * __restrict c, const pthread_condattr_t * __restrict a) {
    if (support_pthread) return __real_pthread_cond_init(c, a);
    return 0;
}
int __wrap_pthread_cond_destroy(pthread_cond_t *c) {
    if (support_pthread) return __real_pthread_cond_destroy(c);
    return 0;
}
int __wrap_pthread_cond_wait(pthread_cond_t * __restrict c, pthread_mutex_t * __restrict m) {
    if (support_pthread) return __real_pthread_cond_wait(c, m);
    return 0;
}
int __wrap___pthread_cond_timedwait(pthread_cond_t * __restrict c, pthread_mutex_t * __restrict m, const struct timespec * __restrict ts) {
    if (support_pthread) return __real___pthread_cond_timedwait(c, m, ts);
    return ETIMEDOUT;
}
int __wrap_pthread_cond_signal(pthread_cond_t *c) {
    if (support_pthread) return __real_pthread_cond_signal(c);
    return 0;
}
int __wrap_pthread_cond_broadcast(pthread_cond_t *c) {
    if (support_pthread) return __real_pthread_cond_broadcast(c);
    return 0;
}

// ---------------------------------------------------------------------------
// fork hooks
// ---------------------------------------------------------------------------

int __wrap_pthread_atfork(void (*prepare)(void), void (*parent)(void), void (*child)(void)) {
    if (support_pthread) return __real_pthread_atfork(prepare, parent, child);
    return 0;
}

// ---------------------------------------------------------------------------
// Read-write locks
// ---------------------------------------------------------------------------

int __wrap_pthread_rwlock_init(pthread_rwlock_t * __restrict rw, const pthread_rwlockattr_t * __restrict a) {
    if (support_pthread) return __real_pthread_rwlock_init(rw, a);
    return 0;
}
int __wrap_pthread_rwlock_destroy(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_destroy(rw);
    return 0;
}
int __wrap_pthread_rwlock_rdlock(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_rdlock(rw);
    return 0;
}
int __wrap_pthread_rwlock_tryrdlock(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_tryrdlock(rw);
    return 0;
}
int __wrap_pthread_rwlock_timedrdlock(pthread_rwlock_t * __restrict rw, const struct timespec * __restrict ts) {
    if (support_pthread) return __real_pthread_rwlock_timedrdlock(rw, ts);
    return 0;
}
int __wrap_pthread_rwlock_wrlock(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_wrlock(rw);
    return 0;
}
int __wrap_pthread_rwlock_trywrlock(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_trywrlock(rw);
    return 0;
}
int __wrap_pthread_rwlock_timedwrlock(pthread_rwlock_t * __restrict rw, const struct timespec * __restrict ts) {
    if (support_pthread) return __real_pthread_rwlock_timedwrlock(rw, ts);
    return 0;
}
int __wrap_pthread_rwlock_unlock(pthread_rwlock_t *rw) {
    if (support_pthread) return __real_pthread_rwlock_unlock(rw);
    return 0;
}

// ---------------------------------------------------------------------------
// Semaphores
// ---------------------------------------------------------------------------

int __wrap_sem_post(sem_t *s) {
    if (support_pthread) return __real_sem_post(s);
    return 0;
}
int __wrap_sem_wait(sem_t *s) {
    if (support_pthread) return __real_sem_wait(s);
    return 0;
}
int __wrap_sem_trywait(sem_t *s) {
    if (support_pthread) return __real_sem_trywait(s);
    errno = EAGAIN;
    return -1;
}

} // extern "C"
