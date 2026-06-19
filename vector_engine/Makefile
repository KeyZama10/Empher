.PHONY: demo clean

demo:
	python3 demo_runner.py

clean:
	@echo "Stopping Docker containers if active..."
	-@docker compose -f deploy/docker-compose.yml down 2>/dev/null || docker-compose -f deploy/docker-compose.yml down 2>/dev/null || true
	@echo "Cleaning up local data and WAL states..."
	rm -rf vector_engine/data/service_discovery_mock*
	rm -rf vector_engine/data/worker_*
	rm -rf vector_db_demo.npz
