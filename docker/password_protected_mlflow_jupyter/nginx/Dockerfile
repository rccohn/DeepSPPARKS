FROM nginx:1.21.6

# change ownership of following paths 
# so that nginx can be run as non-root user
RUN touch "/var/run/nginx.pid"
RUN chown -R nginx:nginx /var/log/nginx \
    && chown -R nginx:nginx /var/run/nginx.pid \
    && chown -R nginx:nginx /etc/nginx \
    && chown -R nginx:nginx /var/cache/nginx

# configuration for reverse-proxy server
COPY --chown=nginx:nginx nginx_conf /etc/nginx/nginx.conf

USER nginx

# without -g "daemon off", nginx will run
# in the background, and the container will
# immediately stop with no active tasks
ENTRYPOINT ["nginx", "-g", "daemon off;"]
