import AppShell from "../_components/AppShell";
import ShrinkageClient from "./shrinkage-client";

export default function ShrinkagePage() {
  return (
    <AppShell crumbs="CAP-CONNECT / Shrinkage">
      <ShrinkageClient />
    </AppShell>
  );
}
