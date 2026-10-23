// Source of structured.js. Rebuild with scripts/bundle-structured.ts.
// The upstream adapter owns task contexts, cancellation and resource journals.
import { instantiateGuest } from "../../../blot2/compiler/guest.ts";

const signature = { parameter: "Array F32", result: "Array F32" } as const;

export async function openStructured(bytes: Uint8Array) {
  const guest = await instantiateGuest(bytes, { asynchronous: true });
  let incoming: PromiseWithResolvers<Float32Array> | undefined;
  let response = Promise.withResolvers<Float32Array>();
  let stopping = false;
  let failure: unknown;
  const capability = guest.capabilityAsync({
    ...signature,
    call: async (values) => {
      if (incoming) throw new Error("Overlapping game frame exchanges");
      const next = Promise.withResolvers<Float32Array>();
      incoming = next;
      response.resolve(values);
      return await next.promise;
    },
  });
  const done = guest.callAsync("main", capability).then(() => {
    if (!stopping) throw new Error("game main returned before shutdown");
  });
  void done.catch((error) => {
    failure = error;
    response.reject(error);
  });
  try {
    const description = await response.promise;
    return {
      description,
      memoryBytes: () => guest.memoryBytes(),
      next(values: Float32Array) {
        if (failure) return Promise.reject(failure);
        if (stopping || !incoming) {
          return Promise.reject(new Error("No pending game frame exchange"));
        }
        response = Promise.withResolvers<Float32Array>();
        const next = incoming;
        incoming = undefined;
        next.resolve(values);
        return response.promise;
      },
      async close() {
        if (stopping) return;
        stopping = true;
        if (!failure && incoming) {
          const next = incoming;
          incoming = undefined;
          next.resolve(new Float32Array([0, 0]));
        }
        try {
          await done;
        } catch { /* The failed request reports the trap. */ }
        // callAsync drains all children before settling. No live stack is retired.
        guest.dispose();
      },
    };
  } catch (error) {
    await done.catch(() => {});
    guest.dispose();
    throw error;
  }
}

// Worker fallback: its frame callback blocks on the existing SAB transport.
// Pure child tasks execute cooperatively, synchronously in this instance.
export async function runStructuredSync(
  bytes: Uint8Array,
  exchange: (values: Float32Array, memory: number) => Float32Array,
) {
  const guest = await instantiateGuest(bytes);
  try {
    const host = guest.capability({
      ...signature,
      call: (values) => exchange(values, guest.memoryBytes()),
    });
    guest.call("main", host);
  } finally {
    guest.dispose();
  }
}
