# Introduction

When setting up a self-hosted DERP (Designated Encrypted Relay for
Packets) server for Tailscale, you might encounter a situation where
port 443 on your host machine is already in use by another service like
Caddy. In this post, I'll explain how to configure Caddy as a reverse
proxy for your DERP server, allowing both services to coexist.

# Caddy configuration

Caddy will handle TLS certificate management automatically, which
simplifies our setup considerably. We'll configure it to reverse proxy
requests to our DERP server running on an alternative port.

The `tls_server_name` directive is crucial here because it tells Caddy
what SNI (Server Name Indication) value to use when connecting to the
backend DERP server. Without this, the TLS handshake would fail as the
backend expects connections for the domain `derp.example.com`.

``` caddy
https://derp.example.com {
    tls {
        key_type rsa2048
    }

    reverse_proxy {
       to https://localhost:8450

       transport http {
            tls_server_name derp.example.com
       }
    }
}
```

# Docker compose setup

Our Docker Compose configuration addresses several important
considerations:

- Port 443 on the host is used by Caddy, so we bind DERP to port 8450
  internally

- We bind port 8450 only to localhost (127.0.0.1) to prevent direct
  access from the internet

- We expose the STUN/TURN UDP port 3478 to facilitate connection
  establishment

- We mount Caddy's automatically generated certificates into the DERP
  container

``` yaml
services:
    derp:
        image: fredliang/derper
        restart: unless-stopped
        ports:
            - 127.0.0.1:8450:443
            - 0.0.0.0:3478:3478/udp
        environment:
            - DERP_DOMAIN=derp.example.com
            - DERP_CERT_MODE=manual
            - DERP_HTTP_PORT=-1
        volumes:
            - /var/lib/caddy/.local/share/caddy/certificates/acme-v02.api.letsencrypt.org-directory/derp.example.com/:/app/certs
```

# Configuring Tailscale to use your DERP server

The final step is to configure Tailscale to route traffic through your
self-hosted DERP server. This is done by adding a custom DERP region in
the Tailscale admin console ACLs: [Tailscale
ACLs](https://login.tailscale.com/admin/acls)

The `OmitDefaultRegions` option controls whether clients connect only to
your self-hosted DERP server or also to Tailscale's official DERP
servers. For initial testing, set this to `true`, and then adjust
according to your needs.

``` json
{
    // ...

    "derpMap": {
        "OmitDefaultRegions": true, // connect only to self-hosted DERP
        "Regions": {
            "900": {
                "RegionID":   900,
                "RegionCode": "myDERP", // custom region code
                "Nodes": [
                    {
                        "Name":     "1",
                        "HostName": "derp.example.com",
                    },
                ],
            },
        },
    },

    // ...
}
```

# Conclusion

With this setup, you now have a self-hosted DERP server for Tailscale
running behind Caddy. This configuration allows you to:

- Use Caddy's automatic TLS certificate management

- Run both services on the same machine

- Control which DERP servers your Tailscale clients connect to

For production environments, you might want to set `OmitDefaultRegions`
to `false` to allow fallback to Tailscale's official DERP servers when
your self-hosted server is unavailable.
