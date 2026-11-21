// ggml-rpc transport between wasm instances in different workers, it replaces llama.cpp/ggml/src/ggml-rpc/transport.cpp
// the port picks a channel (SharedArrayBuffer rings), see "RPC TRANSPORT" in src/workers-code/llama-cpp.js

#include "../llama.cpp/ggml/src/ggml-rpc/transport.h"

#include <emscripten/emscripten.h>
#include <cstdint>
#include <cstring>
#include <memory>
#include <cstdio>
#include <vector>

// pointers and sizes are passed as double: on wasm32, an i32 above 2GB would arrive in JS as a negative number
EM_JS(int, js_rpc_has, (int port), { return _wllama_rpc_has(port); })
EM_JS(int, js_rpc_accept, (int port), { return _wllama_rpc_accept(port); })
EM_JS(int, js_rpc_connect, (int port), { return _wllama_rpc_connect(port); })
EM_JS(int, js_rpc_send, (int port, int conn, double ptr, double size), { return _wllama_rpc_send(port, conn, ptr, size); })
EM_JS(int, js_rpc_recv, (int port, int conn, double ptr, double size), { return _wllama_rpc_recv(port, conn, ptr, size); })
EM_JS(void, js_rpc_close, (int port, int conn), { _wllama_rpc_close(port, conn); })

// ggml-rpc writes message sizes as size_t but reads them as uint64_t, and size_t is 4 bytes on wasm32 (see #340)
// the sender follows the framing to widen them: client sends [cmd 1B][size][input][payload], server sends [size][msg]
enum tx_state
{
    TX_CMD,
    TX_SIZE,
    TX_PAYLOAD,
};

// serialize_graph copies each node pointer with sizeof(uint64_t), so on wasm32 the high half is the next pointer (see #340)
// the graph is: | device (4) | uid (8) | n_nodes (4) | nodes (n_nodes * 8) | ...
static const uint8_t RPC_CMD_GRAPH_COMPUTE = 10;
static std::vector<uint8_t> fix_graph_node_ids(const void *data, size_t size)
{
    std::vector<uint8_t> buf((const uint8_t *)data, (const uint8_t *)data + size);
    uint32_t n_nodes = 0;
    memcpy(&n_nodes, buf.data() + 12, sizeof(n_nodes));
    for (uint32_t i = 0; i < n_nodes; i++)
    {
        memset(buf.data() + 16 + i * 8 + 4, 0, 4);
    }
    return buf;
}

struct socket_t::impl
{
    int port;
    int conn; // 0 for a listening socket
    tx_state tx_start;
    tx_state tx = TX_CMD;
    uint64_t tx_remaining = 0;
    uint8_t tx_cmd = 0;
    bool tx_first_payload = false;
    impl(int port, int conn, bool is_client) : port(port), conn(conn), tx_start(is_client ? TX_CMD : TX_SIZE), tx(tx_start) {}

    bool send_raw(const void *data, size_t size)
    {
        return js_rpc_send(port, conn, (double)(uintptr_t)data, (double)size);
    }
    ~impl()
    {
        if (conn > 0)
            js_rpc_close(port, conn);
    }
};

socket_t::socket_t(std::unique_ptr<impl> p) : pimpl(std::move(p)) {}

socket_t::~socket_t() = default;

bool socket_t::send_data(const void *data, size_t size)
{
    impl &p = *pimpl;
    if (sizeof(size_t) == sizeof(uint64_t))
        return p.send_raw(data, size);
    if (size == 0)
        return true; // empty input of a command
    if (p.tx == TX_CMD && size == 1)
    {
        p.tx = TX_SIZE;
        p.tx_cmd = *(const uint8_t *)data;
        return p.send_raw(data, size);
    }
    if (p.tx == TX_SIZE && size == sizeof(size_t))
    {
        uint64_t v = *(const size_t *)data;
        p.tx_remaining = v;
        p.tx = v > 0 ? TX_PAYLOAD : p.tx_start;
        p.tx_first_payload = true;
        return p.send_raw(&v, sizeof(v));
    }
    if (p.tx == TX_PAYLOAD && size <= p.tx_remaining)
    {
        bool fix_graph = p.tx_start == TX_CMD && p.tx_cmd == RPC_CMD_GRAPH_COMPUTE && p.tx_first_payload && size >= 16;
        p.tx_first_payload = false;
        p.tx_remaining -= size;
        if (p.tx_remaining == 0)
            p.tx = p.tx_start;
        if (fix_graph)
        {
            std::vector<uint8_t> buf = fix_graph_node_ids(data, size);
            return p.send_raw(buf.data(), buf.size());
        }
        return p.send_raw(data, size);
    }
    fprintf(stderr, "@@ERROR@@rpc transport: unexpected send of %zu bytes\n", size);
    return false;
}

bool socket_t::recv_data(void *data, size_t size)
{
    return js_rpc_recv(pimpl->port, pimpl->conn, (double)(uintptr_t)data, (double)size);
}

bool socket_t::flush()
{
    return true;
}

// no RDMA: all-zero caps
void socket_t::get_caps(uint8_t *local_caps)
{
    memset(local_caps, 0, RPC_CONN_CAPS_SIZE);
}

void socket_t::update_caps(const uint8_t *remote_caps)
{
    (void)remote_caps;
}

const char *socket_t::transport_name() const
{
    return "wllama";
}

socket_ptr socket_t::accept()
{
    int conn = js_rpc_accept(pimpl->port);
    if (conn <= 0)
        return nullptr;
    return socket_ptr(new socket_t(std::make_unique<impl>(pimpl->port, conn, false)));
}

socket_ptr socket_t::create_server(const char *host, int port)
{
    (void)host;
    if (!js_rpc_has(port))
        return nullptr;
    return socket_ptr(new socket_t(std::make_unique<impl>(port, 0, false)));
}

socket_ptr socket_t::connect(const char *host, int port)
{
    (void)host;
    int conn = js_rpc_connect(port);
    if (conn <= 0)
        return nullptr;
    return socket_ptr(new socket_t(std::make_unique<impl>(port, conn, true)));
}

bool rpc_transport_init()
{
    return true;
}

void rpc_transport_shutdown() {}
