#import "../../defs.typ": *

#show: post.with(date: "2025-04-12", categories: (
  "chemistry",
  "networks",
))

= Running Tailscale DERP server behind Caddy

When setting up a self-hosted DERP (Designated Encrypted Relay for Packets) server for Tailscale, you might encounter a situation where port 443 on your host machine is already in use by another service like Caddy.
Unfortunately, it's not easy to configure the DERP server behind a reverse proxy, as it does not work via HTTP and requires TLS management to be done manually (see #link("https://github.com/tailscale/tailscale/issues/7745", "this issue")).

#align(center, image("cover.png", width: 50%))

= Caddy configuration

Caddy can handle TLS certificate management automatically. We'll configure it to reverse proxy requests to our DERP server running on an alternative port via HTTPS and to modify the TLS handshake.

The `tls_server_name` is handy here to tell Caddy what SNI (Server Name Indication) value to use when connecting to the backend DERP server. Without this, the TLS handshake would fail as the backend expects connections for the domain `derp.example.com`.

```caddy
https://derp.example.com {
    reverse_proxy {
       to https://localhost:8450

       transport http {
            tls_server_name derp.example.com
       }
    }
}
```

Port 443 on the host is used by Caddy, so we bind the DERP server's port 443 to port 8450 (or any other available) on the host.

Also, it's possible to specify the key type for the TLS certificate. Use this if the default key type is not suitable.

```caddy
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

= Docker compose setup

We bind port 8450 only to localhost (127.0.0.1) to prevent direct access from the internet, expose the STUN/TURN UDP port 3478 to facilitate connection establishment, and mount Caddy's automatically generated certificates into the DERP container for TLS management.

```yaml
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

= Configuring Tailscale to use your DERP server

The final step is to configure Tailscale to route traffic through your self-hosted DERP server. This is done by adding a custom DERP region in the Tailscale #link("https://login.tailscale.com/admin/acls", "admin console ACLs").

The `OmitDefaultRegions` option controls whether clients connect only to the self-hosted DERP server or also to Tailscale's official DERP servers. For initial testing, set this to `true`, and then adjust according to your needs.

```json
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

After testing, you might want to set `OmitDefaultRegions` to `false` to allow fallback to Tailscale's official DERP servers.
