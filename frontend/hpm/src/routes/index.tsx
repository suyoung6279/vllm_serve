import { createBrowserRouter } from "react-router-dom";

import Layout from "../components/layout/Layout";
import MeetingListPage from "../pages/meeting/MeetingListPage";
import MeetingRecordStart from "../pages/meeting/MeetingRecordStart";
import MeetingRecordResult from "../pages/meeting/MeetingRecordResult";
const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      {
        path: "/meeting",
        element: <MeetingListPage />,
      },
      {
        path: "/meeting/:meetingId",
        element: <MeetingRecordStart />,
      },
      {
        path: "/meeting/:meetingId/result",
        element: <MeetingRecordResult />,
      },
    ],
  },
]);

export default router;
